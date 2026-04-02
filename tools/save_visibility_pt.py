# python tools/save_visibility_pt.py \                 #기본틀
#   --config config/building2_visibility.yaml \
#   --iteration 100000 \    #저장할 iteration, -1이면 기본 로딩
#   --split train \
#   --camera_idx 0 \
#   --out debug/visibility_cam0_iter100000.pt

#.pt 저장 스크립트, prefilter_voxel을 이용하여 가시성 정보 계산 후 저장

import os
import yaml
import torch
from argparse import ArgumentParser

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


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="yaml config path")
    parser.add_argument("--iteration", type=int, default=-1, help="불러올 iteration, -1이면 기본 로딩")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--camera_idx", type=int, default=0)
    parser.add_argument("--out", type=str, default="debug/visibility_debug.pt")
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

    if args.camera_idx < 0 or args.camera_idx >= len(cameras):
        raise IndexError(f"camera_idx={args.camera_idx}, num_cameras={len(cameras)}")

    viewpoint = build_viewpoint(lp, args.camera_idx, cameras[args.camera_idx])

    bg_color = [1, 1, 1] if lp.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    with torch.no_grad():
        vis_info = prefilter_voxel(viewpoint, gaussians, pp, background)
        vis_mask = vis_info["mask"]

        selected_ids = torch.nonzero(vis_mask, as_tuple=False).squeeze(1)
        selected_scores = vis_info["score"][vis_mask]
        selected_radii = vis_info["radii"][vis_mask]

        payload = {
            "split": args.split,
            "camera_idx": args.camera_idx,
            "image_name": viewpoint["image_name"],
            "global_ids": selected_ids.detach().cpu(),
            "scores": selected_scores.detach().cpu(),
            "radii": selected_radii.detach().cpu(),
            "xyz": gaussians.get_xyz[vis_mask].detach().cpu(),
            "scaling": gaussians.get_scaling[vis_mask].detach().cpu(),
            "rotation": gaussians.get_rotation[vis_mask].detach().cpu(),
        }

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    torch.save(payload, args.out)

    print(f"saved: {args.out}")
    print(f"split={args.split}")
    print(f"camera_idx={args.camera_idx}")
    print(f"image_name={viewpoint['image_name']}")
    print(f"visible_globals={selected_ids.numel()}")

    if selected_ids.numel() > 0:
        topk = min(10, selected_ids.numel())
        top_vals, local_idx = torch.topk(selected_scores, k=topk)
        top_ids = selected_ids[local_idx]
        top_xyz = gaussians.get_xyz[top_ids].detach().cpu()

        print("\n[top-k visibility globals]")
        for rank in range(topk):
            print(
                f"rank={rank+1:02d} "
                f"global_id={top_ids[rank].item()} "
                f"score={top_vals[rank].item():.6f} "
                f"xyz={top_xyz[rank].tolist()}"
            )


if __name__ == "__main__":
    main()