import os
import sys
import yaml
import torch
from argparse import ArgumentParser

from scene import LargeScene, MixGSModel
from utils.general_utils import parse_cfg
from utils.camera_utils import loadCam
from gaussian_renderer import prefilter_voxel


def build_viewpoint(dataset, camera, idx, scale=1):
    viewpoint_cam = loadCam(dataset, idx, camera, scale)
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


def load_decoder_weights(mixgs, decoder_pth):
    print(f"Loading decoder weights from: {decoder_pth}")
    grid_weight, network_weight = torch.load(decoder_pth, map_location="cuda")
    mixgs.encoder.load_state_dict(grid_weight)
    mixgs.decoder.load_state_dict(network_weight)
    mixgs.encoder.eval()
    mixgs.decoder.eval()


@torch.no_grad()
def measure_counts(cameras, dataset, gaussians, mixgs, pipe, background, split_name="train"):
    decoded_counts = []
    visible_counts = []

    print(f"\n=== Measuring split: {split_name} ({len(cameras)} views) ===")

    for idx, camera in enumerate(cameras):
        viewpoint = build_viewpoint(dataset, camera, idx, scale=1)

        vis_mask = prefilter_voxel(viewpoint, gaussians, pipe, background)
        vis_count = int(vis_mask.sum().item())

        hash_input = [
            gaussians.get_xyz[vis_mask].detach(),
            gaussians.get_scaling[vis_mask].detach(),
            gaussians.get_rotation[vis_mask].detach(),
        ]

        decoded_data = mixgs.step(
            hash_input,
            viewpoint["world_view_transform"][-1, :-1]
        )
        decoded_count = int(decoded_data["d_scaling"].shape[0])

        visible_counts.append(vis_count)
        decoded_counts.append(decoded_count)

        print(
            f"[{split_name:>5}] view {idx:04d} | "
            f"image={viewpoint['image_name']} | "
            f"visible_anchor_count={vis_count} | "
            f"decoded_count={decoded_count}"
        )

    if len(decoded_counts) > 0:
        avg_visible = sum(visible_counts) / len(visible_counts)
        avg_decoded = sum(decoded_counts) / len(decoded_counts)
        total_visible = sum(visible_counts)
        total_decoded = sum(decoded_counts)

        print(f"\n--- {split_name} summary ---")
        print(f"num_views              : {len(decoded_counts)}")
        print(f"avg_visible_count/view : {avg_visible:.2f}")
        print(f"avg_decoded_count/view : {avg_decoded:.2f}")
        print(f"total_visible_count    : {total_visible}")
        print(f"total_decoded_count    : {total_decoded}")

    return visible_counts, decoded_counts


def main():
    parser = ArgumentParser(description="Measure decoded count per view from decoder.pth + point_cloud.ply + config")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--point_cloud", type=str, required=True, help="Path to point_cloud.ply")
    parser.add_argument("--decoder_pth", type=str, required=True, help="Path to decoder.pth")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test", "both"],
        help="Which camera split to measure",
    )
    parser.add_argument("--white_background", action="store_true", help="Override white background")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # parse_cfg usually returns (lp, op, pp)
    # args is passed in so overrides like --config are respected
    lp, op, pp = parse_cfg(cfg, args)

    # Build Gaussian model from config
    modules = __import__("scene")
    model_config = lp.model_config
    gaussians = getattr(modules, model_config["name"])(
        lp.sh_degree, **model_config["kwargs"]
    )

    # Load point cloud
    print(f"Loading point cloud from: {args.point_cloud}")
    gaussians.load_ply(args.point_cloud)

    # Build MixGS model and load decoder weights
    mixgs = MixGSModel(
        hash_args=lp.hash_args,
        net_args=lp.network_args,
    )
    load_decoder_weights(mixgs, args.decoder_pth)

    # Build scene to access cameras
    scene = LargeScene(lp, gaussians)

    bg_color = [1, 1, 1] if (args.white_background or lp.white_background) else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if args.split in ["train", "both"]:
        train_cameras = scene.getTrainCameras()
        measure_counts(train_cameras, lp, gaussians, mixgs, pp, background, split_name="train")

    if args.split in ["test", "both"]:
        test_cameras = scene.getTestCameras()
        measure_counts(test_cameras, lp, gaussians, mixgs, pp, background, split_name="test")


if __name__ == "__main__":
    main()