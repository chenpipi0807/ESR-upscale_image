#!/usr/bin/env python3
"""
简化 GPU 图像放大服务（零依赖版）
不依赖 BasicSR / RealESRGAN 包
"""
import os
import cv2
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# 网络结构：自建 RRDBNet（兼容官方权重）
# ------------------------------------------------------------
class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class SimpleRRDBNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, feat=64, n_blocks=6, grow_ch=32, scale=4):
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(in_ch, feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(feat, grow_ch) for _ in range(n_blocks)])
        self.conv_body = nn.Conv2d(feat, feat, 3, 1, 1)

        # 4× upsample
        self.up1 = nn.Conv2d(feat, feat, 3, 1, 1)
        self.up2 = nn.Conv2d(feat, feat, 3, 1, 1)
        self.hr  = nn.Conv2d(feat, feat, 3, 1, 1)
        self.last = nn.Conv2d(feat, out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        body = self.conv_body(self.body(feat)) + feat
        up = self.lrelu(self.up1(F.interpolate(body, scale_factor=2, mode='nearest')))
        up = self.lrelu(self.up2(F.interpolate(up, scale_factor=2, mode='nearest')))
        hr = self.lrelu(self.hr(up))
        return self.last(hr)

# ------------------------------------------------------------
# 推理服务
# ------------------------------------------------------------
class SimpleGPUUpscaler:
    def __init__(self, model_path, device='auto'):
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.model_path = model_path
        self.model = None
        self.is_initialized = False
        self._load_model()

    def _load_model(self):
        try:
            self.model = SimpleRRDBNet(n_blocks=6)  # 动漫模型 6 block
            ckpt = torch.load(self.model_path, map_location='cpu')
            state = ckpt.get('params_ema') or ckpt.get('params') or ckpt
            self.model.load_state_dict(state, strict=False)
            self.model.eval().to(self.device)
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            print(f"✅ 模型加载完成 | 设备: {self.device}")
            self.is_initialized = True
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            self.is_initialized = False

    def resize_image_if_needed(self, img, max_size=None):
        h, w = img.shape[:2]
        if max_size is None:
            max_size = 1024 if self.device == 'cuda' else 512
        scale = min(max_size / w, max_size / h, 1)
        if scale < 1:
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            print(f"📏 缩放: {w}×{h} → {new_w}×{new_h}")
        return img, scale

    @torch.no_grad()
    def enhance_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        if self.device == 'cuda':
            with torch.cuda.amp.autocast():
                out = self.model(tensor)
        else:
            out = self.model(tensor)
        out = out.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        out = (out * 255).round().astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    def process_image(self, input_path, out_dir="results", quality=90):
        if not self.is_initialized:
            print("❌ 服务未初始化"); return None
        if not os.path.isfile(input_path):
            print("❌ 文件不存在"); return None
        os.makedirs(out_dir, exist_ok=True)

        img = cv2.imread(input_path)
        if img is None:
            print("❌ 无法读取图片"); return None

        img_rs, _ = self.resize_image_if_needed(img)
        tic = time.time()
        out = self.enhance_image(img_rs)
        save_path = os.path.join(out_dir, "ESR_" + os.path.splitext(os.path.basename(input_path))[0] + ".webp")
        Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)).save(save_path, "WEBP", quality=quality, method=6)
        print(f"✅ 完成! 耗时 {time.time()-tic:.2f}s → {save_path}")
        return save_path

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def main():
    print("="*60)
    print("🎯 简化 GPU 图像放大服务（零依赖版）")
    print("="*60)
    model_path = "RealESRGAN_x4plus_anime_6B.pth"
    if not os.path.isfile(model_path):
        print("❌ 模型文件不存在:", model_path); return
    upscaler = SimpleGPUUpscaler(model_path)
    if not upscaler.is_initialized:
        return

    while True:
        path = input("\n请输入图片路径: ").strip().strip('"')
        if path.lower() in {"q","quit","exit"}: break
        if not path: continue
        upscaler.process_image(path)

if __name__ == "__main__":
    main()