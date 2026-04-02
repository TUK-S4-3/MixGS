#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#training 흐름 : 1. prepare_output_and_logger(start_iteration)으로 로그 디렉토리 + logger 생성
#                2. (__init__.py)에서 scene 모델 구성 가져오기, dataset에서 모델 구성 가져오기, gaussian 모델과 MixGS 모델 생성
#                3. (datasets.py)에서 GSDataset과 CacheDataLoader 초기화
#                4. resume/checkpoint 로드 처리
#                5. 반복 학습 진행 : (__init__.py)에서 prefilter_voxel로 가시성 마스크 계산(전역 가우시안 visibility filter) -> (전역 가우시안 -> 디테일 가우시안 : mixgs.step에서 decoded_data 만듬, render_mix에서 결합) MixGS 모델에서 디코딩된 데이터 얻기 -> (__init__.py)에서 render_mix로 렌더링 패키지 얻기 -> 
#                                   (loss_utils.py)에서 손실 계산 -> 그래디언트 계산 -> training_report로 로그 작성, 테스트 및 샘플 보고 -> checkpoint 저장 및 optimizer step + zero_grad 처리 + lr 조정
#
import time
import yaml
import os
import torch
import torchvision
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import prefilter_voxel, render_mix
import sys
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from scene import LargeScene
from scene.mixgs_model_resume import MixGSModel
from scene.datasets import GSDataset, CacheDataLoader
from utils.camera_utils import loadCam
from utils.general_utils import safe_state, parse_cfg
from tqdm import tqdm
from os import makedirs
from utils.image_utils import psnr
from utils.log_utils import tensorboard_log_image, wandb_log_image
from argparse import ArgumentParser, Namespace
from lpipsPyTorch import lpips
from fused_ssim import fused_ssim
from utils.budget_utils import allocate_detail_budget


def build_resume_ckpt_path(model_path, iteration=None):
    if iteration is None:
        return os.path.join(model_path, "resume.pth")
    return os.path.join(model_path, f"resume_{iteration}.pth")


def legacy_full_resume_ckpt_path(model_path, iteration):
    return os.path.join(model_path, f"full_resume_{iteration}.pth")


def resolve_resume_ckpt_path(model_path, iteration):
    candidates = [
        build_resume_ckpt_path(model_path, iteration),
        build_resume_ckpt_path(model_path),
    ]
    if iteration is not None and iteration > 0:
        candidates.append(legacy_full_resume_ckpt_path(model_path, iteration))

    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def save_resume_state(model_path, iteration, gaussians, mixgs):
    state = {
        "format": "mixgs_full_resume_v2",
        "iteration": iteration,
        "gaussians": gaussians.capture(),
        "gaussian_meta": {
            "n_offsets": getattr(gaussians, "n_offsets", None),
            "max_sh_degree": getattr(gaussians, "max_sh_degree", None),
        },
        "mixgs": mixgs.capture(include_optimizer=True),
    }

    latest_path = build_resume_ckpt_path(model_path)
    iter_path = build_resume_ckpt_path(model_path, iteration)
    legacy_path = legacy_full_resume_ckpt_path(model_path, iteration)

    torch.save(state, latest_path)
    torch.save(state, iter_path)
    torch.save(state, legacy_path)

    print(f"[ITER {iteration}] Saved resume checkpoint -> {latest_path}")
    print(f"[ITER {iteration}] Saved iteration resume checkpoint -> {iter_path}")
    print(f"[ITER {iteration}] Saved legacy-compatible resume checkpoint -> {legacy_path}")


def load_resume_state(ckpt_path, gaussians, mixgs, opt):
    state = torch.load(ckpt_path, map_location="cuda")

    if isinstance(state, tuple) and len(state) == 2 and isinstance(state[0], tuple):
        model_params, first_iter = state
        gaussians.restore(model_params, opt)
        print(f"[RESUME] Loaded legacy gaussian-only checkpoint from {ckpt_path}")
        return first_iter

    if not isinstance(state, dict) or "gaussians" not in state:
        raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")

    first_iter = int(state["iteration"])
    gaussians.restore(state["gaussians"], opt)

    if "mixgs" in state:
        mixgs.restore(state["mixgs"], load_optimizer=True)
    else:
        mixgs_state = state.get("mixgs_weights", None)
        if mixgs_state is not None:
            mixgs.restore({
                "encoder": mixgs_state["encoder"] if "encoder" in mixgs_state else mixgs_state.get("grid", {}),
                "decoder": mixgs_state["decoder"],
                "optimizer": state.get("mixgs_optimizer", None),
            }, load_optimizer=("mixgs_optimizer" in state))

    if first_iter >= opt.joint_start_iter:
        gaussians.gaussian_training()

    gaussians.update_learning_rate(first_iter)
    mixgs.update_learning_rate(first_iter)

    print(f"[RESUME] Loaded full checkpoint from {ckpt_path}")
    return first_iter


def training(dataset, opt, pipe, testing_iterations, saving_iterations, refilter_iterations, checkpoint_iterations,
             checkpoint, max_cache_num, debug_from, resume_model_path=None, resume_iteration=-1):
    first_iter = 0
    log_writer, image_logger = prepare_output_and_logger(dataset) #로그 디렉토리 + logger 생성

    modules = __import__('scene') #scene 모듈에서 모델 구성 가져오기
    model_config = dataset.model_config #dataset에서 모델 구성 가져오기
    gaussians = getattr(modules, model_config['name'])(dataset.sh_degree, **model_config['kwargs']) #gaussian 모델 생성

    mixgs = MixGSModel( #MixGS 모델 생성
        hash_args=dataset.hash_args,
        net_args=dataset.network_args,
    )
    mixgs.train_setting(opt)

    resume_from_saved_state = ( #   resume_model_path과 resume_iteration이 주어지고 checkpoint가 없는 경우, full resume 체크포인트에서 재개 시도
        resume_model_path is not None
        and resume_iteration is not None
        and resume_iteration > 0
        and checkpoint is None
    )
    resume_ckpt_path = None
    resume_ckpt_exists = False

    original_model_path = dataset.model_path #dataset의 model_path 저장
    if resume_from_saved_state:
        resume_ckpt_path = resolve_resume_ckpt_path(resume_model_path, resume_iteration)
        resume_ckpt_exists = resume_ckpt_path is not None

        if resume_ckpt_exists:
            scene = LargeScene(dataset, gaussians)
        else:
            dataset.model_path = resume_model_path
            scene = LargeScene(dataset, gaussians, load_iteration=resume_iteration)
            dataset.model_path = original_model_path
            scene.model_path = original_model_path
    else:
        scene = LargeScene(dataset, gaussians)

    gs_dataset = GSDataset(scene.getTrainCameras(), scene, dataset, pipe) #학습 데이터셋 생성, gs_dataset이 비어 있지 않으면 CacheDataLoader로 래핑하여 최대 max_cache_num 이미지를 캐싱
    if len(gs_dataset) > 0:
        print(f"Using maximum cache size of {max_cache_num} for {len(gs_dataset)} training images")
        data_loader = CacheDataLoader(gs_dataset, max_cache_num=max_cache_num, seed=42, batch_size=1, shuffle=True, num_workers=8)

    gaussians.training_setup(opt) #gaussians 모델 학습 설정
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    elif resume_from_saved_state:
        if resume_ckpt_exists:
            first_iter = load_resume_state(resume_ckpt_path, gaussians, mixgs, opt)
        else:
            mixgs.load_weights(resume_model_path, resume_iteration)
            first_iter = resume_iteration

            if resume_iteration >= opt.joint_start_iter:
                gaussians.gaussian_training()

            gaussians.update_learning_rate(resume_iteration)
            mixgs.update_learning_rate(resume_iteration)

            print(f"[RESUME] Loaded point_cloud and decoder from {resume_model_path} at iteration {resume_iteration}")
            print("[RESUME] resume checkpoint not found, so optimizer state is NOT restored.")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0] #배경색 설정, dataset에서 white_background가 true이면 흰색, 아니면 검은색
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda") #배경색을 CUDA 텐서로 변환

    iter_start = torch.cuda.Event(enable_timing=True) 
    iter_end = torch.cuda.Event(enable_timing=True)

    ema_loss_for_log = 0.0
    ema_time_render = 0.0 
    ema_time_loss = 0.0 
    ema_time_densify = 0.0 
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress") 
    first_iter += 1
    iteration = first_iter

    while iteration <= opt.iterations: 
        if len(gs_dataset) == 0:
            print("No training data found")
            print("\n[ITER {}] Saving Gaussians".format(iteration))
            scene.save(iteration, dataset)
            break

        for dataset_index, (cam_info, gt_image) in enumerate(data_loader): #각 iteration마다 데이터 로더에서 카메라 정보와 GT 이미지 가져오기
            iter_start.record() 

            # Render
            start = time.time()
            if (iteration - 1) == debug_from:
                pipe.debug = True

            vis_info = prefilter_voxel(cam_info, gaussians, pipe, background) #vis_mask -> vis_info로 변경하여 가시성 마스크와 점수 모두 반환하도록 수정

            vis_mask = vis_info["mask"]

            if vis_mask.sum() == 0:
                continue

            # iteration마다 가시성 마스크에서 선택된 global의 개수와 상위 10개의 global ID, 점수, 위치 출력하여 가시성 계산이 제대로 되고 있는지 확인
            if iteration % 1000 == 0:
                selected_ids = torch.nonzero(vis_mask, as_tuple=False).squeeze(1)
                selected_scores = vis_info["score"][vis_mask]

                topk = min(10, selected_scores.numel())
                if topk > 0:
                    top_vals, local_idx = torch.topk(selected_scores, k=topk)
                    top_ids = selected_ids[local_idx]
                    top_xyz = gaussians.get_xyz[top_ids].detach().cpu()

                    print(f"\n[ITER {iteration}] top visibility globals")
                    for rank in range(topk):
                        print(
                            f"rank={rank+1:02d} "
                            f"global_id={top_ids[rank].item()} "
                            f"score={top_vals[rank].item():.6f} "
                            f"xyz={top_xyz[rank].tolist()}"
                        )
            if iteration % 5000 == 0:
                selected_ids = torch.nonzero(vis_mask, as_tuple=False).squeeze(1)
                payload = {
                    "iteration": iteration,
                    "global_ids": selected_ids.detach().cpu(),
                    "scores": vis_info["score"][vis_mask].detach().cpu(),
                    "radii": vis_info["radii"][vis_mask].detach().cpu(),
                    "xyz": gaussians.get_xyz[vis_mask].detach().cpu(),
                    "scaling": gaussians.get_scaling[vis_mask].detach().cpu(),
                    "rotation": gaussians.get_rotation[vis_mask].detach().cpu(),
            }

            debug_dir = os.path.join(log_writer.log_dir, "visibility_debug")
            os.makedirs(debug_dir, exist_ok=True)
            torch.save(payload, os.path.join(debug_dir, f"visibility_iter_{iteration}.pt"))

            # hash_input = [gaussians.get_xyz[vis_mask].detach(), gaussians.get_scaling[vis_mask].detach(), gaussians.get_rotation[vis_mask].detach()]
            hash_input = [
                gaussians.get_xyz[vis_mask].detach(),
                gaussians.get_scaling[vis_mask].detach(),
                gaussians.get_rotation[vis_mask].detach()
            ] 

            decoded_data = mixgs.step(hash_input, cam_info['world_view_transform'][-1, :-1])
            if iteration == opt.joint_start_iter:
                gaussians.gaussian_training()

            #추가 
            detail_counts = allocate_detail_budget(
                vis_score=vis_info["score"][vis_mask],
                total_budget=opt.detail_budget_total,
                max_details=gaussians.n_offsets,
                min_details=opt.detail_budget_min,
                temperature=opt.visibility_temperature,
            )

            #  기존 : render_pkg = render_mix(cam_info, gaussians, pipe, background, vis_mask, decoded_data) 
            render_pkg = render_mix(
                cam_info,
                gaussians,
                pipe,
                background,
                vis_info,
                decoded_data,
                detail_counts
            )

            if render_pkg is None:
                continue

            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[
                "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"] #렌더링된 이미지, 뷰 공간 포인트 텐서, 가시성 필터, 반지름 얻기
            end = time.time()
            ema_time_render = 0.4 * (end - start) + 0.6 * ema_time_render  

            # Loss
            start = time.time()
            gt_image = gt_image.cuda() #GT 이미지를 CUDA로 이동
            Ll1 = l1_loss(image, gt_image) #L1 손실 계산, 렌더링된 이미지와 GT 이미지 간의 L1 손실 계산
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))) #총 손실 계산, L1 손실과 DSSIM 손실의 가중 합으로 총 손실 계산

            loss.backward()
            end = time.time()
            ema_time_loss = 0.4 * (end - start) + 0.6 * ema_time_loss #EMA 손실 시간 업데이트, 현재 손실 계산 시간과 이전 EMA 손실 시간으로 EMA 손실 시간 업데이트

            iter_end.record()

            #LOGGING AND CHECKPOINTS
            with torch.no_grad():
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log #EMA 손실 업데이트, 현재 손실과 이전 EMA 손실로 EMA 손실 업데이트
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                grads = gaussians.xyz_gradient_accum / gaussians.denom #그래디언트 계산
                grads[grads.isnan()] = 0.0 
                ema_time = {                    #EMA 시간과 점수, 평균 그래디언트로 ema_time 딕셔너리 생성
                    "render": ema_time_render,
                    "loss": ema_time_loss,
                    "densify": ema_time_densify,
                    "num_points": radii.shape[0],
                    "mean_grad": grads.mean().item(),
                } 
                ema_time["visible_globals"] = int(vis_mask.sum().item())
                ema_time["detail_count_mean"] = float(detail_counts.float().mean().item())
                ema_time["detail_count_sum"] = int(detail_counts.sum().item())
                #학습률 가져오기, gaussians와 mixgs의 optimizer에서 학습률 가져와서 lr 딕셔너리에 저장
                lr = {} 
                for param_group in gaussians.optimizer.param_groups:
                    lr[param_group['name']] = param_group['lr']

                for param_group in mixgs.optimizer.param_groups:
                    lr[param_group['name']] = param_group['lr']

                # Log and save #opt추가
                training_report(dataset, opt, log_writer, image_logger, iteration, Ll1, loss, l1_loss, ema_time, lr,
                                iter_start.elapsed_time(iter_end), testing_iterations, scene, mixgs, (pipe, background))

                if (iteration in saving_iterations):
                    print("\n[ITER {}] Saving Gaussians".format(iteration))
                    # log_writer.log_dir
                    scene.save(iteration, log_writer.log_dir)
                    mixgs.save_weights(log_writer.log_dir, iteration)
                    save_resume_state(log_writer.log_dir, iteration, gaussians, mixgs)

                if (iteration in refilter_iterations):
                    print("\n[ITER {}] Refiltering Training Data".format(iteration))
                    gs_dataset = GSDataset(scene.getTrainCameras(), scene, dataset, pipe)

                # Optimizer step 
                if iteration < opt.iterations:
                    gaussians.optimizer.step()
                    gaussians.update_learning_rate(iteration)
                    mixgs.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)
                    mixgs.optimizer.zero_grad() #MixGS 모델의 optimizer의 그래디언트 초기화
                    mixgs.update_learning_rate(iteration)

                if (iteration in checkpoint_iterations):
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                    save_resume_state(scene.model_path, iteration, gaussians, mixgs)

            iteration += 1
            if iteration >= opt.iterations:
                break


def prepare_output_and_logger(args): #모델 경로 설정, 출력 폴더 생성, logger 구성
    if not args.model_path:
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        # time_stamp = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
        args.model_path = os.path.join("./output/", config_name)
        if args.block_id >= 0:
            if args.block_id < args.block_dim[0] * args.block_dim[1] * args.block_dim[2]:
                args.model_path = f"{args.model_path}/cells/cell{args.block_id}"
                if args.logger_config is not None:
                    args.logger_config['name'] = f"{args.logger_config['name']}_cell{args.block_id}"
            else:
                raise ValueError("Invalid block_id: {}".format(args.block_id))

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # build logger
    log_writer = None
    image_logger = None
    logger_args = {
        "save_dir": args.model_path
    }
    if args.logger_config is None or args.logger_config['logger'] == "tensorboard":
        log_writer = TensorBoardLogger(**logger_args)
        image_logger = tensorboard_log_image
    elif args.logger_config['logger'] == "wandb":
        logger_args.update(name=args.logger_config['name'])
        logger_args.update(project=args.logger_config['project'])
        log_writer = WandbLogger(**logger_args)
        image_logger = wandb_log_image
    else:
        raise ValueError("Unknown logger: {}".format(args.logger_config['logger']))

    return log_writer, image_logger

#opt추가
def training_report(dataset, opt, log_writer, image_logger, iteration, Ll1, loss, l1_loss, ema_time, lr, elapsed,
                    testing_iterations, scene: LargeScene, mixgs, renderArgs): #학습 보고, 로그 작성, 테스트 및 샘플 보고
    if log_writer:
        metrics_to_log = {
            "train_loss_patches/l1_loss": Ll1.item(), 
            "train_loss_patches/total_loss": loss.item(),
            "train_time/render": ema_time["render"],
            "train_time/loss": ema_time["loss"],
            "train_time/densify": ema_time["densify"],
            "train_time/num_points": ema_time["num_points"],
            "train_time/mean_grad": ema_time["mean_grad"],
            #visibility 관련 지표 추가
            #현재 뷰에서 몇 개의 global이 보이는지, 실제 detail budget이 얼마나 분배 되었는지
            "train_time/visible_globals": ema_time["visible_globals"],
            "train_time/detail_count_mean": ema_time["detail_count_mean"],
            "train_time/detail_count_sum": ema_time["detail_count_sum"],
            "iter_time": elapsed,
        }
        for key, value in lr.items(): 
            metrics_to_log["trainer/" + key] = value
        log_writer.log_metrics(metrics_to_log, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations: 
        torch.cuda.empty_cache() 
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()}, # 검증 구성 정의, 테스트 카메라와 훈련 카메라에서 선택된 카메라로 구성된 검증 구성을 포함하는 튜플 생성
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs: #각 검증 구성에 대해, 카메라가 정의되어 있으면 각 카메라에 대해 렌더링된 이미지와 GT 이미지를 얻고 L1 손실, PSNR, SSIM, LPIPS 계산, 로그 작성
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssims_test = 0.0
                lpips_test = 0.0
                valid_count = 0 #추가 - 유효한 카메라 수 계산을 위한 변수 초기화

                for idx, camera in enumerate(config['cameras']):
                    viewpoint_cam = loadCam(dataset, idx, camera, 1)

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
                    org_img = viewpoint_cam.original_image 

                    #training loop에서와 동일한 렌더링 과정 수행
                    # vis_mask = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs) 
                    vis_info = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs)
                    vis_mask = vis_info["mask"]

                    

                    if vis_mask.sum() == 0:
                        continue
                    valid_count += 1


                    # hash_input = [scene.gaussians.get_xyz[vis_mask].detach(), scene.gaussians.get_scaling[vis_mask].detach(), 
                    #               scene.gaussians.get_rotation[vis_mask].detach()]
                
                    hash_input = [
                        scene.gaussians.get_xyz[vis_mask].detach(),
                        scene.gaussians.get_scaling[vis_mask].detach(),
                        scene.gaussians.get_rotation[vis_mask].detach()
                    ]

                    decoded_data = mixgs.step(hash_input, viewpoint['world_view_transform'][-1, :-1]) 
                    
                    #추가
                    detail_counts = allocate_detail_budget(
                        vis_score=vis_info["score"][vis_mask],
                        total_budget=opt.detail_budget_total,
                        max_details=scene.gaussians.n_offsets,
                        min_details=opt.detail_budget_min,
                        temperature=opt.visibility_temperature,
                    )

                    # render_pkg = render_mix(viewpoint, scene.gaussians, *renderArgs, vis_mask, decoded_data) 
                    render_pkg = render_mix(
                    viewpoint,
                    scene.gaussians,
                    *renderArgs,
                    vis_info,
                    decoded_data,
                    detail_counts
                    )

                    #추가
                    if render_pkg is None:
                        valid_count -= 1
                        continue


                    image = torch.clamp(render_pkg["render"], 0.0, 1.0) #렌더링된 이미지 클램프, 렌더링된 이미지를 0과 1 사이로 클램프하여 유효한 이미지 값 보장
                    gt_image = torch.clamp(org_img.to("cuda"), 0.0, 1.0) #GT 이미지 클램프, GT 이미지를 CUDA로 이동하고 0과 1 사이로 클램프하여 유효한 이미지 값 보장

                    if log_writer and (idx < 5):
                        grid = torchvision.utils.make_grid(torch.concat([image, gt_image], dim=-1))
                        image_logger(
                            log_writer=log_writer,
                            tag=config['name'] + "_view_{}".format(viewpoint["image_name"]),
                            image_tensor=grid,
                            step=iteration,
                        )
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssims_test += ssim(image, gt_image).mean().double()
                    lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()

                if valid_count > 0:
                    psnr_test /= valid_count
                    l1_test /= valid_count
                    ssims_test /= valid_count
                    lpips_test /= valid_count
                else:
                    continue
                #원래는 len(config['cameras'])로 나누었지만, 유효한 카메라 수로 나누도록 수정하여 유효한 카메라가 없는 경우를 처리하도록 변경
                # psnr_test /= len(config['cameras'])
                # l1_test /= len(config['cameras'])
                # ssims_test /= len(config['cameras'])
                # lpips_test /= len(config['cameras'])

                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], l1_test, psnr_test, ssims_test, lpips_test))
                if log_writer:
                    metrics_to_log = {
                        config['name'] + '/loss_viewpoint/l1_loss': l1_test,
                        config['name'] + '/loss_viewpoint/psnr': psnr_test,
                        config['name'] + '/loss_viewpoint/ssim': ssims_test,
                        config['name'] + '/loss_viewpoint/lpips': lpips_test,
                    }
                    log_writer.log_metrics(metrics_to_log, iteration)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--config', type=str, help='train config file path')
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6007)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--block_id', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)

    parser.add_argument("--test_iterations", nargs="+", type=int, default=[200_000, 250_000, 300_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[2000, 10000, 18000, 28000, 40000, 55000, 70000, 100_000,150_000, 250_000, 300_000])

    parser.add_argument("--refilter_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--resume_model_path", type=str, default=None)
    parser.add_argument("--resume_iteration", type=int, default=-1)
    parser.add_argument("--max_cache_num", type=int, default=32)
    args = parser.parse_args(sys.argv[1:])
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        lp, op, pp = parse_cfg(cfg, args)
        args.save_iterations.append(op.iterations)

    print("Optimizing " + lp.model_path)

    import resource
    resource.setrlimit(resource.RLIMIT_NOFILE, [11264, 65535])

    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp, op, pp, args.test_iterations, args.save_iterations, args.refilter_iterations,
             args.checkpoint_iterations, args.start_checkpoint, args.max_cache_num, args.debug_from,
             args.resume_model_path, args.resume_iteration)

    # All done
    print("\nTraining complete.")