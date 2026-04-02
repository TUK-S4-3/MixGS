# python tools/export_visibility_ply.py \
#   --config config/building2_visibility.yaml \
#   --iteration 100000 \
#   --split train \
#   --camera_idx 0 \
#   --top_k 300 \     #상위 300개 글로벌만 ply로 저장
#   --out debug/visibility_top300_cam0_iter100000.ply

# python tools/export_visibility_ply.py \
#   --config config/building2_visibility.yaml \
#   --iteration 100000 \
#   --split train \
#   --camera_idx 0 \
#   --top_k 0.01 \   #상위 1% 글로벌만 ply로 저장
#   --out debug/visibility_top300_cam0_iter100000.ply

#.ply 뷰어 저장

import os
import yaml
import torch
import numpy as np
from argparse import ArgumentParser
from plyfile import PlyData, PlyElement

from utils.general_utils import parse_cfg, safe_state
from gaussian_renderer import prefilter_voxel
from scene import LargeScene
from utils.camera_utils import loadCam


def build_viewpoint(dataset, camera_idx, camera):
    viewpoint_cam = loadCam(dataset, camera_idx, camera, 1)
    viewpoint = {
        "FoVx": viewpoint_cam.FoVx,
        "FoVy": viewpoint_cam.FoVy,
        "image_name": viewpoint_cam.image_name,
        "image_height": viewpoint_cam.image_height,
        "image_width": viewpoint_cam.image_width,
        "camera_center": viewpoint_cam.camera_center,
        "world_view_transform": viewpoint_cam.world_view_transform,
        "full_proj_transform": viewpoint_cam.full_proj_transform,
    }
    return viewpoint


def write_point_ply(xyz, colors, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    vertex_data = np.empty(
        xyz.shape[0],
        dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ],
    )
    vertex_data["x"] = xyz[:, 0]
    vertex_data["y"] = xyz[:, 1]
    vertex_data["z"] = xyz[:, 2]
    vertex_data["red"] = colors[:, 0]
    vertex_data["green"] = colors[:, 1]
    vertex_data["blue"] = colors[:, 2]

    ply = PlyData([PlyElement.describe(vertex_data, "vertex")], text=False)
    ply.write(out_path)


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--camera_idx", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_ratio", type=float, default=0.0)
    parser.add_argument("--min_score", type=float, default=-1.0)
    parser.add_argument("--out", type=str, default="debug/visibility_selected.ply")

    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--block_id", type=int, default=-1)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6007)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--refilter_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--resume_model_path", type=str, default=None)
    parser.add_argument("--resume_iteration", type=int, default=-1)
    parser.add_argument("--max_cache_num", type=int, default=32)

    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    lp, op, pp = parse_cfg(cfg, args)
    safe_state(args.quiet)

    modules = __import__("scene")
    model_config = lp.model_config
    gaussians = getattr(modules, model_config["name"])(lp.sh_degree, **model_config["kwargs"])

    if args.iteration >= 0:
        scene = LargeScene(lp, gaussians, load_iteration=args.iteration)
    else:
        scene = LargeScene(lp, gaussians)

    cameras = scene.getTrainCameras() if args.split == "train" else scene.getTestCameras()
    if len(cameras) == 0:
        raise RuntimeError(f"{args.split} cameras is empty")

    if not (0 <= args.camera_idx < len(cameras)):
        raise IndexError(f"camera_idx={args.camera_idx}, num_cameras={len(cameras)}")

    viewpoint = build_viewpoint(lp, args.camera_idx, cameras[args.camera_idx])

    bg_color = [1, 1, 1] if lp.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    with torch.no_grad():
        vis_info = prefilter_voxel(viewpoint, gaussians, pp, background)
        vis_mask = vis_info["mask"]

        selected_ids = torch.nonzero(vis_mask, as_tuple=False).squeeze(1)
        scores = vis_info["score"][vis_mask]
        xyz = gaussians.get_xyz[vis_mask]

        if selected_ids.numel() == 0:
            raise RuntimeError("No visible globals for this camera")

        if args.top_k > 0:
            k = min(args.top_k, selected_ids.numel())
            keep_local = torch.topk(scores, k=k).indices
        elif args.top_ratio > 0:
            k = max(1, int(selected_ids.numel() * args.top_ratio))
            keep_local = torch.topk(scores, k=k).indices
        elif args.min_score >= 0:
            keep_local = torch.nonzero(scores >= args.min_score, as_tuple=False).squeeze(1)
            if keep_local.numel() == 0:
                raise RuntimeError("No globals satisfy min_score")
        else:
            keep_local = torch.arange(selected_ids.numel(), device=scores.device)

        keep_ids = selected_ids[keep_local]
        keep_scores = scores[keep_local]
        keep_xyz = xyz[keep_local].detach().cpu().numpy()

    s = keep_scores.detach().cpu().numpy()
    s_min, s_max = float(s.min()), float(s.max())
    norm = (s - s_min) / (s_max - s_min + 1e-8)

    colors = np.stack([
        (norm * 255).astype(np.uint8),                    # red
        np.zeros_like(norm, dtype=np.uint8),             # green
        ((1.0 - norm) * 255).astype(np.uint8),           # blue
    ], axis=1)

    write_point_ply(keep_xyz, colors, args.out)

    txt_path = os.path.splitext(args.out)[0] + "_meta.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"split={args.split}\n")
        f.write(f"camera_idx={args.camera_idx}\n")
        f.write(f"image_name={viewpoint['image_name']}\n")
        f.write(f"num_exported={keep_ids.numel()}\n")
        for gid, sc in zip(keep_ids.detach().cpu().tolist(), keep_scores.detach().cpu().tolist()):
            f.write(f"{gid}\t{sc:.8f}\n")

    print(f"saved ply: {args.out}")
    print(f"saved meta: {txt_path}")
    print(f"exported globals: {keep_ids.numel()}")


if __name__ == "__main__":
    main()