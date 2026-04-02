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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel, GatheredGaussian
from utils.sh_utils import eval_sh
from utils.large_utils import in_frustum

#성능 최적화용 visible / invisible만 알 수 있었음 -> 얼마나 잘보이는지 vis_mask와 vis_score로 구분하여 반환하도록 수정
def prefilter_voxel(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
                    override_color=None): 
    """
    Render the scene.                               렌더 성능 / split 처리 등 처리용 
    Background tensor (bg_color) must be on GPU!
    """
    from diff_gaussian_rasterization_filter import GaussianRasterizationSettings, GaussianRasterizer
    tanfovx = math.tan(viewpoint_camera["FoVx"] * 0.5) 
    tanfovy = math.tan(viewpoint_camera["FoVy"] * 0.5) 

    #GaussianRasterizationSettings 클래스의 인스턴스를 생성하여 raster_settings 변수에 저장. 이 클래스는 가우시안 렌더링을 위한 설정을 포함함
    raster_settings = GaussianRasterizationSettings(     
        image_height=int(viewpoint_camera["image_height"]),
        image_width=int(viewpoint_camera["image_width"]), 
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera["world_view_transform"],
        projmatrix=viewpoint_camera["full_proj_transform"],
        sh_degree=1,
        campos=viewpoint_camera["camera_center"],
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings) 

    means3D = pc.get_xyz

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    #수정 필요 현재 v_i = max(r_i, 0) -> v_i = log(1+r_i)로 변경하여 값이 너무 뾰족하지 않도록 변경 필요 -> 수정 완료
    radii_pure = rasterizer.visible_filter(means3D=means3D,                 
                                           scales=scales[:, :3],
                                           rotations=rotations,
                                           cov3D_precomp=cov3D_precomp)
    
    vis_mask = radii_pure > 0              
    #vis_score = torch.colmap(radii_pure) #radius
    vis_score = torch.log1p(torch.clamp(radii_pure, min=0.0)) #log_radius사용

    return {"mask": vis_mask, "score": vis_score, "radii": radii_pure}

#vis_mask -> vis_info로 변경하여 mask와 score 모두 반환하도록 수정, detail counts 추가
def render_mix(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, vis_info, decoded_data, detail_counts, scaling_modifier=1.0): # 기존 gaussian에 decoder(디테일 가우시안)에서 추출한 수정 값 반영해 혼합 렌더링
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    # #vis_mask = 원본 가우시안 ori_xyz, ori_rot 가져오기
    # #decoded_data = 디테일 카우시안의 크기, 회전, 색상, 불투명도 변화 가져오기
    #ori_xyz = pc.get_xyz[vis_mask].detach() 
    #ori_rot = pc.get_rotation[vis_mask].detach() 
   
    vis_mask = vis_info["mask"]
    if vis_mask.sum() == 0:
        return None

    #디테일 수정 값
    d_scaling = decoded_data["d_scaling"].to(torch.float32) 
    d_rotation = decoded_data["d_rotation"].to(torch.float32) 
    d_sh = decoded_data["d_color"].to(torch.float32)
    d_opacity = decoded_data["d_opacity"].to(torch.float32)

    
    base_xyz = pc.get_xyz[vis_mask].detach()            # [V, 3]
    base_rot = pc.get_rotation[vis_mask].detach()       # [V, 4]
    #base_scale = pc.get_scaling[vis_mask].detach()      # [V, 3]
    #base_opacity = pc.get_opacity[vis_mask]             # [V, 1]
    base_offsets = pc.get_offset[vis_mask]              # [V, K, 3]

    K = base_offsets.shape[1]

    detail_counts = detail_counts.to(device=base_xyz.device, dtype=torch.long)
    detail_counts = torch.clamp(detail_counts, min=0, max=K)

    slot_ids = torch.arange(K, device=base_xyz.device).unsqueeze(0)   # [1, K]
    slot_mask = slot_ids < detail_counts.unsqueeze(1)                 # [V, K]

    detail_xyz = (base_xyz.unsqueeze(1) + base_offsets)[slot_mask]    # [sum(K_i), 3]
    detail_rot = pc.rotation_activation(base_rot.unsqueeze(1) + d_rotation)[slot_mask]
    detail_scale = torch.clamp_min(d_scaling, pipe.scale_min)[slot_mask]
    detail_opacity = d_opacity[slot_mask]
    detail_color = torch.sigmoid(d_sh)[slot_mask]

    #shape와 opacity activation 여부 확인
    V = base_xyz.shape[0]
    assert d_scaling.shape[:2] == (V, K)
    assert d_rotation.shape[:2] == (V, K)
    assert d_sh.shape[:2] == (V, K)
    assert d_opacity.shape[:2] == (V, K)

    # 전역 가우시안과 디테일 가우시안 1대1 가정 부분 제거
    #num = len(d_scaling)
    #means3D = ori_xyz + pc.get_offset[vis_mask].reshape(num, -1) #기존 가우시안 주변에 디테일 가우시안의 위치 변화(offset)를 더하여 최종 3D 위치를 계산하여 means3D 변수에 저장
    #큰 구조 = 전역 가우시안 + 작은 구조 = 디테일 가우시안의 위치 변화(offset) -> means3D

    # 원본 가우시안은 sh 기반 시점 의존 색을 유지, 디테일 가우시안은 디코더가 직접 예측한 색을 사용하여 색상 사전 계산
    pc_features = pc.get_features[vis_mask].transpose(1, 2)                             
    shs_view = pc_features.view(pc_features.shape[0], -1, (pc.max_sh_degree + 1) ** 2)
    dir_pp = (pc.get_xyz[vis_mask] - viewpoint_camera["camera_center"].repeat(pc_features.shape[0], 1))
    dir_norm = dir_pp.norm(dim=1, keepdim=True).clamp_min(1e-6)   #dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True) -> dir_pp_normalized = dir_pp / dir_norm, dir_norm이 0이 되는 경우를 방지하기 위해 clamp_min 추가
    dir_pp_normalized = dir_pp / dir_norm
    sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized) 
    colors_precomp_base = torch.clamp_min(sh2rgb + 0.5, 0.0) #color_precomp -> colors_precomp_base

    # colors_precomp = torch.cat([torch.sigmoid(d_sh), colors_precomp_base], dim=0)  
    # opacity = d_opacity

    #스케일과 회전도 디테일 가우시안의 변화(d_scaling, d_rotation)를 적용하여 최종 스케일과 회전을 계산
    # rotations = pc.rotation_activation(ori_rot + d_rotation)   
    # res_scales = torch.clamp_min(d_scaling, pipe.scale_min) 

    ori_means3D = pc.get_xyz[vis_mask]     
    ori_opacity = pc.get_opacity[vis_mask] 
    ori_scales = pc.get_scaling[vis_mask]  
    ori_rotations = pc.get_rotation[vis_mask] 

    #원본 가우시안과 디테일 가우시안의 데이터를 연결하여 최종 렌더링에 사용할 means3D, opacity, scales, rotations 계산
    # means3D = torch.cat([means3D, ori_means3D], dim=0)        
    # opacity = torch.cat([opacity, ori_opacity], dim=0)       
    # scales = torch.cat([res_scales, ori_scales], dim=0)       
    # rotations = torch.cat([rotations, ori_rotations], dim=0)  
    means3D = torch.cat([detail_xyz, ori_means3D], dim=0)
    opacity = torch.cat([detail_opacity, ori_opacity], dim=0)
    scales = torch.cat([detail_scale, ori_scales], dim=0)
    rotations = torch.cat([detail_rot, ori_rotations], dim=0)
    colors_precomp = torch.cat([detail_color, colors_precomp_base], dim=0)

    #렌더링 직전
    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0 #screenspace_points 변수를 2D 대응 텐서를 만들고 gradient를 계산할 수 있도록 설정하여 초기화. 이 텐서는 가우시안의 3D 위치를 화면 공간으로 변환한 결과를 저장하는 데 사용됨
    try:
         #screenspace_points 텐서에 대해 그래디언트를 유지하도록 설정하여, 렌더링 과정에서 이 텐서에 대한 그래디언트가 계산되고 저장될 수 있도록 함. 이렇게 하면 나중에 역전파 단계에서 이 텐서에 대한 그래디언트를 사용할 수 있음
        screenspace_points.retain_grad()       
    except:
        pass

    tanfovx = math.tan(viewpoint_camera["FoVx"] * 0.5)  
    tanfovy = math.tan(viewpoint_camera["FoVy"] * 0.5)  

    #GaussianRasterizationSettings 클래스의 인스턴스를 생성하여 raster_settings 변수에 저장. 이 클래스는 가우시안 렌더링을 위한 설정을 포함함
    raster_settings = GaussianRasterizationSettings(    
        image_height=int(viewpoint_camera["image_height"]),
        image_width=int(viewpoint_camera["image_width"]),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera["world_view_transform"],
        projmatrix=viewpoint_camera["full_proj_transform"],
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera["camera_center"],
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=False,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings) #

    means2D = screenspace_points

    cov3D_precomp = None

     # rasterizer 인스턴스를 호출하여 가우시안 렌더링을 수행하고, 렌더링된 이미지(rendered_image), 각 가우시안의 반지름(radii), 깊이 이미지(depth_image)를 반환하여 저장
    rendered_image, radii, depth_image = rasterizer(
        means3D=means3D,    
        means2D=means2D,
        shs=None, #shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)       

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth": depth_image,
        "scale": scales,
    }


