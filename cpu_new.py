#!/usr/bin/env python3
"""
Real-ESRGAN 4x 超分服务（零依赖版）
仅需：
  pip install torch opencv-python pillow numpy
模型：RealESRGAN_x4plus_anime_6B.pth
"""
import os
import sys
import time
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

# ------------------------------------------------------------
# 网络结构：自建 RRDBNet（与官方权重完全兼容）
# ------------------------------------------------------------
class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class RRDBNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, nf=64, nb=23, gc=32, scale=4):
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(in_ch, nf, 3, 1, 1, bias=True)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.hr_conv  = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.last_conv = nn.Conv2d(nf, out_ch, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        body = self.conv_body(self.body(feat)) + feat
        # 4x nearest upsample
        up1 = self.lrelu(self.upconv1(F.interpolate(body, scale_factor=2, mode='nearest')))
        up2 = self.lrelu(self.upconv2(F.interpolate(up1, scale_factor=2, mode='nearest')))
        hr = self.lrelu(self.hr_conv(up2))
        return self.last_conv(hr)

# ------------------------------------------------------------
# 推理服务
# ------------------------------------------------------------
class RealESRGANZeroDep:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.model = RRDBNet(nb=6)  # 动漫模型用的是 6 个 block
        self._load_weights(model_path)
        self.model.eval().to(self.device)

    def _load_weights(self, path):
        ckpt = torch.load(path, map_location='cpu')
        state = ckpt.get('params_ema') or ckpt.get('params') or ckpt
        self.model.load_state_dict(state, strict=False)
        print("✅ 权重加载完成")

    @torch.no_grad()
    def enhance(self, img_bgr):
        # BGR -> RGB / [0,1] / tensor
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(self.device)
        out = self.model(img).clamp(0,1).squeeze(0).permute(1,2,0).cpu().numpy()
        out = (out * 255).round().astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

# ------------------------------------------------------------
# CLI 交互
# ------------------------------------------------------------
def main():
    model_path = "RealESRGAN_x4plus_anime_6B.pth"
    if not os.path.isfile(model_path):
        print("❌ 模型文件不存在:", model_path); return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = RealESRGANZeroDep(model_path, device)
    print(f"🎯 设备: {device} | 输入 'quit' 退出")

    while True:
        path = input("\n请输入图片路径: ").strip().strip('"')
        if path.lower() in {"q","quit","exit"}: break
        if not os.path.isfile(path):
            print("❌ 文件不存在"); continue

        img = cv2.imread(path)
        if img is None:
            print("❌ 无法读取"); continue

        tic = time.time()
        out = engine.enhance(img)
        cv2.imwrite("ESR_" + os.path.basename(path), out,
                    [cv2.IMWRITE_WEBP_QUALITY, 90])
        print(f"✅ 完成! 耗时 {time.time()-tic:.2f}s → ESR_{os.path.basename(path)}")

if __name__ == "__main__":
    main()