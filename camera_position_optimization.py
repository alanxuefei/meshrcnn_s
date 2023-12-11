import numpy as np
import torch
import torch.nn as nn
from pytorch3d.renderer import (
    look_at_rotation, 
    FoVPerspectiveCameras, 
    BlendParams, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer, 
    SoftSilhouetteShader
)
from PIL import Image

class Model(nn.Module):
    def __init__(self, meshes, image_ref, device, renderer=None):
        super().__init__()
        self.meshes = meshes
        self.device = meshes.device

        # Initialize a perspective camera.
        cameras = FoVPerspectiveCameras(device=device)

        # Set blending parameters.
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

        # Define rasterization settings.
        raster_settings = RasterizationSettings(
            image_size=256, 
            blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
            faces_per_pixel=100
        )

        # Create a silhouette mesh renderer as the default renderer.
        self.renderer = renderer if renderer is not None else MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            shader=SoftSilhouetteShader(blend_params=blend_params)
        )

        # Prepare the silhouette of the reference RGB image.
        image_ref = torch.from_numpy((image_ref[..., :3].max(-1) != 1).astype(np.float32))
        self.register_buffer('image_ref', image_ref)

        # Create an optimizable parameter for the camera position.
        self.camera_position = nn.Parameter(
            torch.from_numpy(np.array([0.0, -1.0, 3.0], dtype=np.float32)).to(device))

    def forward(self):
        # Calculate rotation and translation based on camera position.
        R = look_at_rotation(self.camera_position[None, :], device=self.device)
        T = -torch.bmm(R.transpose(1, 2), self.camera_position[None, :, None])[:, :, 0]

        # Render the image using the renderer.
        image = self.renderer(meshes_world=self.meshes.clone(), R=R, T=T)

        # Calculate the silhouette loss.
        loss = torch.sum((image[..., 3] - self.image_ref) ** 2)
        return loss, image
