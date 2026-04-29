import os
import torch
from torchvision.utils import save_image
from torch.utils.data import Dataset
from torchvision import datasets
from utils.general_utils import PILtoTorch
from PIL import Image
import numpy as np

class CameraDataset(Dataset):
    
    def __init__(self, viewpoint_stack, white_background):
        self.viewpoint_stack = viewpoint_stack
        self.bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
        
    def __getitem__(self, index):
        viewpoint_cam = self.viewpoint_stack[index]
        if viewpoint_cam.meta_only:
            with Image.open(viewpoint_cam.image_path) as image_load:
                im_data = np.array(image_load.convert("RGBA"))
            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + self.bg * (1 - norm_data[:, :, 3:4])
            image_load = Image.fromarray(np.array(arr*255.0, dtype=np.uint8), "RGB")
            resized_image_rgb = PILtoTorch(image_load, viewpoint_cam.resolution)
            viewpoint_image = resized_image_rgb[:3, ...].clamp(0.0, 1.0)
            if resized_image_rgb.shape[1] == 4:
                gt_alpha_mask = resized_image_rgb[3:4, ...]
                viewpoint_image *= gt_alpha_mask
            else:
                viewpoint_image *= torch.ones((1, viewpoint_cam.image_height, viewpoint_cam.image_width))
        else:
            viewpoint_image = viewpoint_cam.image

        # ====== ADD: load binary mask from sibling "masks/" and set gt_alpha_mask ======
        # images/.../xxx.png -> masks/.../xxx.png に置き換え
        mask_path = viewpoint_cam.image_path
        mask_path = mask_path.replace(f"{os.sep}images{os.sep}", f"{os.sep}masks{os.sep}")

        if os.path.exists(mask_path):
            with Image.open(mask_path) as m:
                m = m.convert("L")  # 0..255
                m_np = np.array(m, dtype=np.uint8)

            # 2値（0/1）にする：255を前景、0を背景として扱う
            # （あなたのmasksは2値なのでこれでOK）
            m_bin = (m_np >= 128).astype(np.float32)  # [H,W] in {0,1}

            # torch: [1,H,W]
            viewpoint_cam.gt_alpha_mask = torch.from_numpy(m_bin).unsqueeze(0)

        else:
            # masksが無い場合は None のまま（opa_maskを使うなら masks を必ず用意）
            viewpoint_cam.gt_alpha_mask = getattr(viewpoint_cam, "gt_alpha_mask", None)
        # ====== ADD END ======

        return viewpoint_image, viewpoint_cam
    
    def __len__(self):
        return len(self.viewpoint_stack)
    
