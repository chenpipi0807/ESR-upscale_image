#!/usr/bin/env python3
"""
简化的GPU图像放大服务 - 完全独立实现
不依赖BasicSR和RealESRGAN包，直接使用PyTorch
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

class ResidualDenseBlock(nn.Module):
    """残差密集块"""
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    """残差中的残差密集块"""
    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class SimpleRRDBNet(nn.Module):
    """简化的RRDBNet网络"""
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4):
        super(SimpleRRDBNet, self).__init__()
        self.scale = scale
        
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.ModuleList([RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        # 上采样层
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        
        body_feat = feat
        for block in self.body:
            body_feat = block(body_feat)
        
        feat = feat + self.conv_body(body_feat)
        
        # 4倍上采样
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_hr(feat))
        out = self.conv_last(feat)
        
        return out

class SimpleGPUUpscaler:
    """简化的GPU图像放大服务"""
    
    def __init__(self, model_path, device='auto'):
        # 设备选择
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
                print(f"🎮 检测到CUDA设备: {torch.cuda.get_device_name(0)}")
                print(f"💾 GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            else:
                self.device = 'cpu'
                print("⚠️  未检测到CUDA设备，使用CPU模式")
        else:
            self.device = device
            
        self.model_path = model_path
        self.model = None
        self.is_initialized = False
        
        print("🚀 正在初始化简化GPU放大服务...")
        self.initialize_model()
        
    def initialize_model(self):
        """初始化模型"""
        start_time = time.time()
        
        try:
            # 检查模型文件
            if not os.path.exists(self.model_path):
                print(f"❌ 模型文件不存在: {self.model_path}")
                return False
            
            # 创建模型
            self.model = SimpleRRDBNet(
                num_in_ch=3, 
                num_out_ch=3, 
                num_feat=64, 
                num_block=6, 
                num_grow_ch=32, 
                scale=4
            )
            
            # 加载权重
            print("📦 加载模型权重...")
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # 尝试不同的权重键
            if 'params_ema' in checkpoint:
                state_dict = checkpoint['params_ema']
            elif 'params' in checkpoint:
                state_dict = checkpoint['params']
            else:
                state_dict = checkpoint
            
            # 过滤不匹配的键
            model_dict = self.model.state_dict()
            filtered_dict = {}
            
            for k, v in state_dict.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    filtered_dict[k] = v
                else:
                    print(f"⚠️  跳过不匹配的权重: {k}")
            
            # 加载过滤后的权重
            self.model.load_state_dict(filtered_dict, strict=False)
            self.model.eval()
            self.model = self.model.to(self.device)
            
            # GPU优化
            if self.device == 'cuda':
                torch.backends.cudnn.benchmark = True
                # 预热
                dummy_input = torch.randn(1, 3, 64, 64).to(self.device)
                with torch.no_grad():
                    _ = self.model(dummy_input)
                del dummy_input
                torch.cuda.empty_cache()
                print("🔥 GPU预热完成")
            
            load_time = time.time() - start_time
            print(f"✅ 模型加载完成！耗时: {load_time:.2f}秒")
            print(f"📱 设备: {self.device}")
            if self.device == 'cuda':
                print(f"💾 GPU显存使用: {torch.cuda.memory_allocated() / 1024**2:.1f}MB")
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {str(e)}")
            return False
    
    def resize_image_if_needed(self, image, max_size=None):
        """根据GPU显存调整图像大小"""
        h, w = image.shape[:2]
        
        if max_size is None:
            if self.device == 'cuda':
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if gpu_memory_gb >= 8:
                    max_size = 1024
                elif gpu_memory_gb >= 6:
                    max_size = 768
                else:
                    max_size = 512
            else:
                max_size = 512
        
        scale_factor = min(max_size / w, max_size / h)
        
        if scale_factor < 1:
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            print(f"📏 图像缩放: {w}×{h} → {new_w}×{new_h} (最大处理: {max_size}px)")
        
        return image, scale_factor
    
    def enhance_image(self, img):
        """增强图像"""
        try:
            # OpenCV读取的是BGR格式，需要转换为RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 转换为tensor
            img_tensor = img_rgb.astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(np.transpose(img_tensor, (2, 0, 1))).float()
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            
            # 推理
            with torch.no_grad():
                if self.device == 'cuda':
                    with torch.cuda.amp.autocast():
                        output = self.model(img_tensor)
                else:
                    output = self.model(img_tensor)
            
            # 转换回numpy (RGB格式)
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output, (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)
            
            # 转换回BGR格式供OpenCV使用
            output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            
            # 清理GPU缓存
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            
            return output_bgr
            
        except Exception as e:
            print(f"❌ 图像增强失败: {e}")
            return None
    
    def process_image(self, input_path, output_dir="results", quality=90):
        """处理图像"""
        if not self.is_initialized:
            print("❌ 服务未初始化！")
            return None
            
        start_time = time.time()
        
        try:
            # 检查输入
            if not os.path.exists(input_path):
                print(f"❌ 文件不存在: {input_path}")
                return None
            
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成输出文件名
            imgname = os.path.splitext(os.path.basename(input_path))[0]
            output_path = os.path.join(output_dir, f"ESR-{imgname}.webp")
            
            print(f"📸 处理图像: {input_path}")
            
            # 加载图像
            load_start = time.time()
            img = cv2.imread(input_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"❌ 无法加载图像: {input_path}")
                return None
            
            original_shape = img.shape
            load_time = time.time() - load_start
            
            # 调整大小
            resize_start = time.time()
            img_resized, scale_factor = self.resize_image_if_needed(img)
            resized_shape = img_resized.shape
            resize_time = time.time() - resize_start
            
            # 增强图像
            enhance_start = time.time()
            output_img = self.enhance_image(img_resized)
            enhance_time = time.time() - enhance_start
            
            if output_img is None:
                return None
            
            output_shape = output_img.shape
            
            # 保存图像
            save_start = time.time()
            # output_img已经是BGR格式，转换为RGB用于PIL保存
            output_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(output_rgb)
            pil_img.save(output_path, "WEBP", quality=quality, method=6)
            save_time = time.time() - save_start
            
            # 总时间
            total_time = time.time() - start_time
            
            # 输出结果
            print(f"✅ 处理完成!")
            print(f"   📥 输入: {original_shape[1]}×{original_shape[0]} → {resized_shape[1]}×{resized_shape[0]}")
            print(f"   📤 输出: {output_shape[1]}×{output_shape[0]} (4倍放大)")
            print(f"   💾 保存为: {output_path}")
            print(f"   🎨 质量: {quality}%")
            if self.device == 'cuda':
                print(f"   🎮 GPU显存峰值: {torch.cuda.max_memory_allocated() / 1024**2:.1f}MB")
            print(f"   ⏱️  时长分析:")
            print(f"      - 图像加载: {load_time:.3f}s")
            print(f"      - 图像调整: {resize_time:.3f}s")
            print(f"      - AI增强: {enhance_time:.3f}s")
            print(f"      - 文件保存: {save_time:.3f}s")
            print(f"      - 总计: {total_time:.3f}s")
            
            # 重置显存统计
            if self.device == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            
            return output_path
            
        except Exception as e:
            error_time = time.time() - start_time
            print(f"❌ 处理失败 (耗时{error_time:.2f}s): {str(e)}")
            return None

def main():
    """主服务循环"""
    print("=" * 60)
    print("🎯 简化GPU图像放大服务")
    print("=" * 60)
    
    # 配置
    model_path = "RealESRGAN_x4plus_anime_6B.pth"
    
    # 检查模型文件
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请确保模型文件在当前目录中")
        return
    
    # 初始化服务
    service = SimpleGPUUpscaler(model_path=model_path, device='auto')
    
    if not service.is_initialized:
        print("❌ 服务初始化失败！")
        return
    
    print("\n" + "=" * 60)
    print("🎉 服务已启动！")
    print("💡 提示: 输入 'quit' 或 'exit' 退出服务")
    print("💡 默认设置: 4倍放大，90%质量WebP格式")
    print("=" * 60)
    
    # 服务循环
    processed_count = 0
    total_processing_time = 0
    
    while True:
        try:
            print(f"\n📂 已处理 {processed_count} 张图片")
            input_path = input("请输入图片路径: ").strip()
            
            if input_path.lower() in ['quit', 'exit', 'q']:
                print(f"\n👋 服务结束！共处理了 {processed_count} 张图片")
                if processed_count > 0:
                    avg_time = total_processing_time / processed_count
                    print(f"📊 平均处理时间: {avg_time:.3f}s/张")
                break
            
            if not input_path:
                print("❌ 请输入有效的图片路径")
                continue
            
            input_path = input_path.strip('"\'')
            
            start_time = time.time()
            result = service.process_image(input_path, "results", 90)
            process_time = time.time() - start_time
            
            if result:
                processed_count += 1
                total_processing_time += process_time
                
        except KeyboardInterrupt:
            print(f"\n\n👋 用户中断！共处理了 {processed_count} 张图片")
            break
        except Exception as e:
            print(f"❌ 发生错误: {str(e)}")

if __name__ == "__main__":
    main()
