# import torch
# import math
# from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
# from scene.gaussian_model import GaussianModel, GatheredGaussian
# from utils.sh_utils import eval_sh
# from utils.large_utils import in_frustum


# def prefilter_voxel(
#     viewpoint_camera,
#     pc: GaussianModel,
#     pipe,
#     bg_color: torch.Tensor,
#     scaling_modifier=1.0,
#     override_color=None,
# ):
#     """
#     Render the scene.

#     Background tensor (bg_color) must be on GPU!
#     """
#     from diff_gaussian_rasterization_filter import GaussianRasterizationSettings, GaussianRasterizer

#     tanfovx = math.tan(viewpoint_camera["FoVx"] * 0.5)
#     tanfovy = math.tan(viewpoint_camera["FoVy"] * 0.5)

#     raster_settings = GaussianRasterizationSettings(
#         image_height=int(viewpoint_camera["image_height"]),
#         image_width=int(viewpoint_camera["image_width"]),
#         tanfovx=tanfovx,
#         tanfovy=tanfovy,
#         bg=bg_color,
#         scale_modifier=scaling_modifier,
#         viewmatrix=viewpoint_camera["world_view_transform"],
#         projmatrix=viewpoint_camera["full_proj_transform"],
#         sh_degree=1,
#         campos=viewpoint_camera["camera_center"],
#         prefiltered=False,
#         debug=pipe.debug,
#     )

#     rasterizer = GaussianRasterizer(raster_settings=raster_settings)

#     means3D = pc.get_xyz

#     scales = None
#     rotations = None
#     cov3D_precomp = None
#     if pipe.compute_cov3D_python:
#         cov3D_precomp = pc.get_covariance(scaling_modifier)
#     else:
#         scales = pc.get_scaling
#         rotations = pc.get_rotation

#     radii_pure = rasterizer.visible_filter(
#         means3D=means3D,
#         scales=scales[:, :3],
#         rotations=rotations,
#         cov3D_precomp=cov3D_precomp,
#     )

#     return radii_pure > 0


# def render_mix(
#     viewpoint_camera,
#     pc: GaussianModel,
#     pipe,
#     bg_color: torch.Tensor,
#     vis_mask,
#     decoded_data,
#     expanded_idx,
#     scaling_modifier=1.0,
# ):
#     """
#     Render the scene with:
#     - decoded/detail gaussians from expanded visible anchors
#     - original visible gaussians

#     Args:
#         vis_mask: boolean mask over all global gaussians
#         decoded_data: dict with d_scaling, d_rotation, d_color, d_opacity
#         expanded_idx: long tensor of shape (num_decoded,), indexing into visible anchors
#     """
#     # visible anchors
#     vis_xyz = pc.get_xyz[vis_mask]
#     vis_rot = pc.get_rotation[vis_mask]
#     vis_offset = pc.get_offset[vis_mask]
#     vis_features = pc.get_features[vis_mask]
#     vis_opacity = pc.get_opacity[vis_mask]
#     vis_scaling = pc.get_scaling[vis_mask]

#     # expanded anchors for decoded/detail gaussians
#     anchor_xyz = vis_xyz[expanded_idx].detach()
#     anchor_rot = vis_rot[expanded_idx].detach()
#     anchor_offset = vis_offset[expanded_idx]

#     d_scaling = decoded_data["d_scaling"].to(torch.float32)
#     d_rotation = decoded_data["d_rotation"].to(torch.float32)
#     d_sh = decoded_data["d_color"].to(torch.float32)
#     d_opacity = decoded_data["d_opacity"].to(torch.float32)

#     num_decoded = d_scaling.shape[0]
#     assert anchor_xyz.shape[0] == num_decoded, (
#         f"expanded anchor count ({anchor_xyz.shape[0]}) "
#         f"must match decoded gaussian count ({num_decoded})"
#     )

#     # decoded gaussian centers = expanded anchor xyz + corresponding offset
#     means3D_decoded = anchor_xyz + anchor_offset.reshape(num_decoded, -1)

#     # decoded colors
#     decoded_colors = torch.sigmoid(d_sh)

#     # decoded rotations / scales / opacities
#     decoded_rotations = pc.rotation_activation(anchor_rot + d_rotation)
#     scale_min = getattr(pipe, "scale_min", 0.002)
#     decoded_scales = torch.clamp_min(d_scaling, scale_min)
#     decoded_opacity = d_opacity

#     # original visible anchor colors
#     pc_features = vis_features.transpose(1, 2)
#     shs_view = pc_features.view(pc_features.shape[0], -1, (pc.max_sh_degree + 1) ** 2)
#     dir_pp = vis_xyz - viewpoint_camera["camera_center"].repeat(pc_features.shape[0], 1)
#     dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
#     sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
#     original_colors = torch.clamp_min(sh2rgb + 0.5, 0.0)

#     # concatenate decoded + original visible anchors
#     means3D = torch.cat([means3D_decoded, vis_xyz], dim=0)
#     colors_precomp = torch.cat([decoded_colors, original_colors], dim=0)
#     opacity = torch.cat([decoded_opacity, vis_opacity], dim=0)
#     scales = torch.cat([decoded_scales, vis_scaling], dim=0)
#     rotations = torch.cat([decoded_rotations, vis_rot], dim=0)

#     screenspace_points = (
#         torch.zeros_like(
#             means3D,
#             dtype=means3D.dtype,
#             requires_grad=True,
#             device="cuda",
#         )
#         + 0
#     )
#     try:
#         screenspace_points.retain_grad()
#     except Exception:
#         pass

#     tanfovx = math.tan(viewpoint_camera["FoVx"] * 0.5)
#     tanfovy = math.tan(viewpoint_camera["FoVy"] * 0.5)

#     raster_settings = GaussianRasterizationSettings(
#         image_height=int(viewpoint_camera["image_height"]),
#         image_width=int(viewpoint_camera["image_width"]),
#         tanfovx=tanfovx,
#         tanfovy=tanfovy,
#         bg=bg_color,
#         scale_modifier=scaling_modifier,
#         viewmatrix=viewpoint_camera["world_view_transform"],
#         projmatrix=viewpoint_camera["full_proj_transform"],
#         sh_degree=pc.active_sh_degree,
#         campos=viewpoint_camera["camera_center"],
#         prefiltered=False,
#         debug=pipe.debug,
#         antialiasing=False,
#     )
#     rasterizer = GaussianRasterizer(raster_settings=raster_settings)

#     means2D = screenspace_points
#     cov3D_precomp = None

#     rendered_image, radii, depth_image = rasterizer(
#         means3D=means3D,
#         means2D=means2D,
#         shs=None,
#         colors_precomp=colors_precomp,
#         opacities=opacity,
#         scales=scales,
#         rotations=rotations,
#         cov3D_precomp=cov3D_precomp,
#     )

#     return {
#         "render": rendered_image,
#         "viewspace_points": screenspace_points,
#         "visibility_filter": radii > 0,
#         "radii": radii,
#         "depth": depth_image,
#         "scale": decoded_scales,
#     }