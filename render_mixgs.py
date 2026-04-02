#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import yaml
import json
import torch
import torchvision
import time
from tqdm import tqdm
from scene import LargeScene
from scene.mixgs_model_resume import MixGSModel
from scene.datasets import GSDataset
from os import makedirs
from gaussian_renderer import prefilter_voxel, render_mix
from utils.general_utils import safe_state, parse_cfg, colorize
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from utils.budget_utils import allocate_detail_budget


def render_set(model_path, name, iteration, gs_dataset, gaussians, mixgs, opt, pipeline, background):
    avg_render_time = 0.0
    max_render_time = 0.0
    avg_memory = 0.0
    max_memory = 0.0
    processed_frames = 0

    render_path = os.path.join(model_path, name, f"ours_{iteration}", "renders")
    gts_path = os.path.join(model_path, name, f"ours_{iteration}", "gt")
    depth_path = os.path.join(model_path, name, f"ours_{iteration}", "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    data_loader = DataLoader(gs_dataset, batch_size=1, shuffle=False, num_workers=0)

    for idx, (cam_info, gt_image) in enumerate(tqdm(data_loader, desc=f"Rendering {name}")):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.time()

        vis_info = prefilter_voxel(cam_info, gaussians, pipeline, background)
        vis_mask = vis_info["mask"]

        if vis_mask.sum() == 0:
            continue

        hash_input = [
            gaussians.get_xyz[vis_mask].detach(),
            gaussians.get_scaling[vis_mask].detach(),
            gaussians.get_rotation[vis_mask].detach(),
        ]

        decoded_data = mixgs.step(hash_input, cam_info["world_view_transform"][0][-1, :-1])

        detail_counts = allocate_detail_budget(
            vis_score=vis_info["score"][vis_mask],
            total_budget=opt.detail_budget_total,
            max_details=gaussians.n_offsets,
            min_details=opt.detail_budget_min,
            temperature=opt.visibility_temperature,
        )

        render_pkg = render_mix(
            cam_info,
            gaussians,
            pipeline,
            background,
            vis_info,
            decoded_data,
            detail_counts,
        )

        if render_pkg is None:
            continue

        torch.cuda.synchronize()
        end = time.time()

        rendering = render_pkg["render"]
        depth = render_pkg["depth"]

        depth_vis = colorize(depth.cpu().squeeze(0), cmap_name="jet")
        gt = gt_image.squeeze(0)[0:3, :, :]

        frame_time = end - start
        avg_render_time += frame_time
        max_render_time = max(max_render_time, frame_time)

        forward_max_memory_allocated = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
        avg_memory += forward_max_memory_allocated
        max_memory = max(max_memory, forward_max_memory_allocated)

        torchvision.utils.save_image(rendering, os.path.join(render_path, f"{idx:05d}.png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, f"{idx:05d}.png"))
        torchvision.utils.save_image(
            depth_vis.permute(2, 0, 1).unsqueeze(0),
            os.path.join(depth_path, f"{idx:05d}.png"),
        )

        processed_frames += 1

    if processed_frames == 0:
        print(f"[WARN] No valid frames rendered for split: {name}")
        stats = {
            "Average FPS": 0.0,
            "Min FPS": 0.0,
            "Average Memory(M)": 0.0,
            "Max Memory(M)": 0.0,
            "Number of Gaussians": int(gaussians.get_xyz.shape[0]),
            "Processed Frames": 0,
        }
    else:
        stats = {
            "Average FPS": processed_frames / avg_render_time,
            "Min FPS": 1.0 / max_render_time if max_render_time > 0 else 0.0,
            "Average Memory(M)": avg_memory / processed_frames,
            "Max Memory(M)": max_memory,
            "Number of Gaussians": int(gaussians.get_xyz.shape[0]),
            "Processed Frames": processed_frames,
        }

    with open(os.path.join(model_path, f"costs_{name}.json"), "w") as fp:
        json.dump(stats, fp, indent=2)

    print(f"\n[{name}] Processed Frames: {stats['Processed Frames']}")
    print(f"[{name}] Average FPS: {stats['Average FPS']:.4f}")
    print(f"[{name}] Min FPS: {stats['Min FPS']:.4f}")
    print(f"[{name}] Average Memory: {stats['Average Memory(M)']:.4f} M")
    print(f"[{name}] Max Memory: {stats['Max Memory(M)']:.4f} M")
    print(f"[{name}] Number of Gaussians: {stats['Number of Gaussians']}")


def render_sets(dataset, opt, iteration, pipeline, load_vq, skip_train, skip_test, custom_test):
    with torch.no_grad():
        modules = __import__("scene")
        model_config = dataset.model_config
        gaussians = getattr(modules, model_config["name"])(dataset.sh_degree, **model_config["kwargs"])

        if custom_test:
            dataset.source_path = custom_test
            filename = os.path.basename(dataset.source_path)

        scene = LargeScene(dataset, gaussians, load_iteration=iteration, load_vq=load_vq, shuffle=False)

        mixgs = MixGSModel(
            hash_args=dataset.hash_args,
            net_args=dataset.network_args,
        )
        mixgs.load_weights(dataset.model_path, iteration)

        print(f"Number of Gaussians: {gaussians.get_xyz.shape[0]}")

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if custom_test:
            views = scene.getTrainCameras() + scene.getTestCameras()
            gs_dataset = GSDataset(views, scene, dataset, pipeline)
            render_set(dataset.model_path, filename, scene.loaded_iter, gs_dataset, gaussians, mixgs, opt, pipeline, background)
            print("Skip both train and test, render all views")
        else:
            if not skip_train:
                gs_dataset = GSDataset(scene.getTrainCameras(), scene, dataset, pipeline)
                render_set(dataset.model_path, "train", scene.loaded_iter, gs_dataset, gaussians, mixgs, opt, pipeline, background)

            if not skip_test:
                gs_dataset = GSDataset(scene.getTestCameras(), scene, dataset, pipeline)
                render_set(dataset.model_path, "test", scene.loaded_iter, gs_dataset, gaussians, mixgs, opt, pipeline, background)


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--config", type=str, help="train config file path of fused model")
    parser.add_argument("--model_path", type=str, help="model path of fused model")
    parser.add_argument("--custom_test", type=str, help="appointed test path")
    parser.add_argument("--load_vq", action="store_true")
    parser.add_argument("--block_id", type=int, default=-1)
    parser.add_argument("--iteration", default=300000, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    if args.model_path is None:
        args.model_path = os.path.join("output", os.path.basename(args.config).split(".")[0])

    import resource
    resource.setrlimit(resource.RLIMIT_NOFILE, [11264, 65535])

    safe_state(args.quiet)

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        lp, op, pp = parse_cfg(cfg, args)

    render_sets(lp, op, args.iteration, pp, args.load_vq, args.skip_train, args.skip_test, args.custom_test)