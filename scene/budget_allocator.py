import torch
import torch.nn.functional as F


def _normalize01(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if x.numel() == 0:
        return x
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min + eps)


class BudgetAllocator:
    def __init__(
        self,
        min_count: int = 1,
        max_count: int = 3,
        vis_weight: float = 0.6,
        edge_weight: float = 0.4,
    ):
        self.min_count = min_count
        self.max_count = max_count
        self.vis_weight = vis_weight
        self.edge_weight = edge_weight

    @torch.no_grad()
    def compute_importance(
        self,
        viewpoint_camera,
        pc,
        vis_mask: torch.Tensor,
        gt_image: torch.Tensor,
        pipe,
        bg_color: torch.Tensor,
    ) -> torch.Tensor:
        vis_xyz = pc.get_xyz[vis_mask]
        if vis_xyz.shape[0] == 0:
            return torch.empty(0, device=pc.get_xyz.device)

        device = vis_xyz.device

        # gt_image expected shape: (3, H, W)
        if gt_image.dim() == 4:
            gt_image = gt_image.squeeze(0)
        gt_image = gt_image.to(device)

        # 1) visibility score: inverse distance
        cam_center = viewpoint_camera["camera_center"].to(device).reshape(-1)[:3]
        dist = torch.norm(vis_xyz - cam_center[None, :], dim=1)
        vis_score = 1.0 / (dist + 1e-6)

        # 2) edge/detail map from GT
        gt_gray = (
            0.2989 * gt_image[0] +
            0.5870 * gt_image[1] +
            0.1140 * gt_image[2]
        ).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

        sobel_x = torch.tensor(
            [[[-1, 0, 1],
              [-2, 0, 2],
              [-1, 0, 1]]],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)  # (1,1,3,3)

        sobel_y = torch.tensor(
            [[[-1, -2, -1],
              [ 0,  0,  0],
              [ 1,  2,  1]]],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)  # (1,1,3,3)

        grad_x = F.conv2d(gt_gray, sobel_x, padding=1)
        grad_y = F.conv2d(gt_gray, sobel_y, padding=1)
        edge_map = torch.sqrt(grad_x ** 2 + grad_y ** 2).squeeze(0).squeeze(0)  # (H,W)

        # 3) project visible anchors to image plane
        ones = torch.ones((vis_xyz.shape[0], 1), device=device)
        xyz_h = torch.cat([vis_xyz, ones], dim=1)  # (N,4)

        full_proj = viewpoint_camera["full_proj_transform"].to(device)
        if full_proj.dim() == 3:
            full_proj = full_proj.squeeze(0)  # (4,4)

        clip = xyz_h @ full_proj  # (N,4)
        ndc = clip[:, :3] / (clip[:, 3:4] + 1e-8)  # (N,3)

        H = int(viewpoint_camera["image_height"])
        W = int(viewpoint_camera["image_width"])

        px = ((ndc[:, 0] + 1.0) * 0.5 * (W - 1)).round().long()
        py = ((1.0 - ndc[:, 1]) * 0.5 * (H - 1)).round().long()

        px = px.clamp(0, W - 1)
        py = py.clamp(0, H - 1)

        edge_score = edge_map[py, px]

        # 4) weighted importance
        vis_score = _normalize01(vis_score).reshape(-1)
        edge_score = _normalize01(edge_score).reshape(-1)

        score = self.vis_weight * vis_score + self.edge_weight * edge_score
        return score

    @torch.no_grad()
    def allocate(self, importance: torch.Tensor, target_budget: int) -> torch.Tensor:
        num_visible = importance.shape[0]
        if num_visible == 0:
            return torch.empty(0, dtype=torch.long, device=importance.device)

        min_required = self.min_count * num_visible
        max_allowed = self.max_count * num_visible

        target_budget = max(target_budget, min_required)
        target_budget = min(target_budget, max_allowed)

        alloc = torch.full(
            (num_visible,),
            fill_value=self.min_count,
            dtype=torch.long,
            device=importance.device,
        )

        extra = target_budget - min_required
        if extra <= 0:
            return alloc

        order = torch.argsort(importance, descending=True)

        add_second = min(num_visible, extra)
        alloc[order[:add_second]] += 1
        extra -= add_second

        if extra > 0:
            add_third = min(num_visible, extra)
            alloc[order[:add_third]] += 1

        return alloc

    @torch.no_grad()
    def expand_indices(self, alloc_counts: torch.Tensor) -> torch.Tensor:
        if alloc_counts.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=alloc_counts.device)

        idx = torch.arange(
            alloc_counts.shape[0],
            device=alloc_counts.device,
            dtype=torch.long,
        )
        return torch.repeat_interleave(idx, alloc_counts)