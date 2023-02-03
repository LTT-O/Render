import argparse
import os
import torchvision

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj
import torch
# Data structures and functions for rendering
from pytorch3d.structures import Meshes
import numpy as np
from tqdm import tqdm
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    blending,
    HardFlatShader,
    TexturesVertex
)

import sys
import os


from pytorch3d.ops import sample_points_from_meshes
sys.path.append(os.path.abspath(''))


def tensor2video(tensor,gray=False):
    video = tensor.detach().cpu().numpy()
    video = video*255.
    video = np.maximum(np.minimum(video, 255), 0)
    if not gray:
        video = video.transpose(0,2,3,1) #[:,:,[2,1,0]]

    return video.astype(np.uint8).copy()
def obj2image(obj_filename, img_size=512, focal=1200, znear=5, zfar=15, cameras_dis=6, cuda=True):
    if cuda:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    mesh = load_objs_as_meshes([obj_filename], device=device, load_textures=False)

    R, T = look_at_view_transform(dist=cameras_dis, elev=2, azim=0)
    fov = 2 * np.arctan(img_size // 2 / focal) * 180. / np.pi
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=znear, zfar=zfar, fov=fov)

    raster_settings = RasterizationSettings(
        image_size=img_size, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )

    lights = PointLights(device=device, location=[[0.0, 0.0, 10]])

    blend_params = blending.BlendParams(background_color=[0, 0, 0])
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
                device=device,
                cameras=cameras,
                lights=lights,
                blend_params=blend_params
            )
    )
    texture = TexturesVertex(torch.ones_like(mesh._verts_list[0].unsqueeze(0)))
    mesh.textures = texture
    images = renderer(mesh)[...,:3].permute(0, 3, 1, 2)

    return images


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_path', type=str, required=True, help='the path of obj that need to be render')
    parser.add_argument('--save_path', default="./demo.png", type=str, help='render image save path')
    parser.add_argument('--cameras_dis', default=6.0, type=float, help='You may need to make changes as appropriate')

    args = parser.parse_args()

    path = args.obj_path
    save_path = args.save_path
    # for demo1.obj, cameras_dis can be set to 1.0
    # for demo2.obj, cameras_dis can be set to 6.0
    image = obj2image(path, cameras_dis=args.cameras_dis)
    torchvision.utils.save_image(image, save_path)