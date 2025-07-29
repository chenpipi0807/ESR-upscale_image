import os
import time
import cv2
import numpy as np
import torch
from PIL import Image
from datetime import datetime
import sys

class RealESRGANService:
    """Real-ESRGAN Service - 预加载模型，快速处理图片"""
    
    def __init__(self, model_path, device='cpu', scale=4):
        self.device = device
        self.scale = scale
        self.model_path = model_path
        self.model = None
        self.is_initialized = False
        
        print("🚀 正在初始化 Real-ESRGAN 服务...")
        self.initialize_model()
        
    def initialize_model(self):
        """初始化并缓存模型"""
        start_time = time.time()
        
        try:
            # Import RRDBNet from basicsr
            from basicsr.archs.rrdbnet_arch import RRDBNet
            
            # Initialize model architecture
            self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                               num_block=6, num_grow_ch=32, scale=4)
            
            # Load pretrained weights
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if 'params_ema' in checkpoint:
                self.model.load_state_dict(checkpoint['params_ema'])
            elif 'params' in checkpoint:
                self.model.load_state_dict(checkpoint['params'])
            else:
                self.model.load_state_dict(checkpoint)
                
            self.model.eval()
            self.model = self.model.to(self.device)
            
            load_time = time.time() - start_time
            print(f"✅ 模型加载完成！耗时: {load_time:.2f}秒")
            print(f"📱 设备: {self.device}")
            print(f"🎯 放大倍数: {self.scale}x")
            self.is_initialized = True
            
        except ImportError:
            print("📦 正在安装 BasicSR...")
            os.system("pip install basicsr")
            self.initialize_model()  # 递归调用重新初始化
            
        except Exception as e:
            print(f"❌ 模型加载失败: {str(e)}")
            sys.exit(1)
    
    def resize_image_keep_aspect_ratio(self, image, max_size=512):
        """调整图像大小，保持宽高比"""
        h, w = image.shape[:2]
        
        # 计算缩放因子
        scale_factor = min(max_size / w, max_size / h)
        
        # 只有当图像大于max_size时才缩放
        if scale_factor < 1:
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        return image, scale_factor
    
    def enhance(self, img):
        """使用Real-ESRGAN增强图像"""
        # 转换为tensor
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        img = img.unsqueeze(0).to(self.device)
        
        # 推理
        with torch.no_grad():
            output = self.model(img)
        
        # 转换回numpy
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output, (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        
        return output
    
    def process_image(self, input_path, compression_quality="lossless"):
        """处理单张图像"""
        if not self.is_initialized:
            print("❌ 服务未初始化！")
            return None
            
        start_time = time.time()
        
        try:
            # 检查输入文件
            if not os.path.exists(input_path):
                print(f"❌ 文件不存在: {input_path}")
                return None
            
            # 生成输出文件名
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            output_file = f"ESR-{base_name}.webp"
            
            print(f"📸 处理图像: {input_path}")
            
            # 加载图像
            load_start = time.time()
            img = cv2.imread(input_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"❌ 无法加载图像: {input_path}")
                return None
            
            original_shape = img.shape
            load_time = time.time() - load_start
            
            # 调整图像大小
            resize_start = time.time()
            img_resized, scale_factor = self.resize_image_keep_aspect_ratio(img, 512)
            resized_shape = img_resized.shape
            resize_time = time.time() - resize_start
            
            # 放大图像
            upscale_start = time.time()
            output_img = self.enhance(img_resized)
            upscale_time = time.time() - upscale_start
            
            output_shape = output_img.shape
            
            # 保存图像
            save_start = time.time()
            output_img_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(output_img_rgb)
            
            # 根据压缩质量设置保存参数
            if compression_quality == "lossless":
                pil_img.save(output_file, "WEBP", lossless=True, method=6)
                quality_info = "无损压缩"
            elif compression_quality == "90":
                pil_img.save(output_file, "WEBP", quality=90, method=6)
                quality_info = "90%质量"
            else:
                pil_img.save(output_file, "WEBP", quality=95, method=6)
                quality_info = "95%质量"
                
            save_time = time.time() - save_start
            
            # 计算总时间
            total_time = time.time() - start_time
            
            # 输出结果
            print(f"✅ 处理完成!")
            print(f"   📥 输入: {original_shape[1]}×{original_shape[0]} → {resized_shape[1]}×{resized_shape[0]}")
            print(f"   📤 输出: {output_shape[1]}×{output_shape[0]} ({quality_info})")
            print(f"   💾 保存为: {output_file}")
            print(f"   ⏱️  时长分析:")
            print(f"      - 图像加载: {load_time:.3f}s")
            print(f"      - 图像调整: {resize_time:.3f}s") 
            print(f"      - AI放大: {upscale_time:.3f}s")
            print(f"      - 文件保存: {save_time:.3f}s")
            print(f"      - 总计: {total_time:.3f}s")
            
            return output_file
            
        except Exception as e:
            error_time = time.time() - start_time
            print(f"❌ 处理失败 (耗时{error_time:.2f}s): {str(e)}")
            return None

def get_compression_choice():
    """获取用户的压缩选择"""
    print("\n🎨 选择WebP压缩质量:")
    print("1. 无损压缩 (文件较大，质量最佳)")
    print("2. 90%质量 (文件适中，质量很好)")
    
    while True:
        choice = input("请选择 (1/2): ").strip()
        if choice == "1":
            return "lossless"
        elif choice == "2":
            return "90"
        else:
            print("❌ 无效选择，请输入 1 或 2")

def main():
    """主服务循环"""
    print("=" * 60)
    print("🎯 Real-ESRGAN 图像放大服务")
    print("=" * 60)
    
    # 配置
    model_path = "RealESRGAN_x4plus_anime_6B.pth"
    device = "cpu"
    
    # 检查模型文件
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请确保模型文件在当前目录中")
        return
    
    # 获取压缩质量选择
    compression_quality = get_compression_choice()
    
    # 初始化服务
    service = RealESRGANService(model_path=model_path, device=device)
    
    print("\n" + "=" * 60)
    print("🎉 服务已启动！模型已缓存在内存中")
    print("💡 提示: 输入 'quit' 或 'exit' 退出服务")
    print("=" * 60)
    
    # 服务循环
    processed_count = 0
    total_processing_time = 0
    
    while True:
        try:
            # 获取用户输入
            print(f"\n📂 已处理 {processed_count} 张图片")
            input_path = input("请输入图片路径: ").strip()
            
            # 检查退出命令
            if input_path.lower() in ['quit', 'exit', 'q']:
                print(f"\n👋 服务结束！共处理了 {processed_count} 张图片")
                if processed_count > 0:
                    avg_time = total_processing_time / processed_count
                    print(f"📊 平均处理时间: {avg_time:.3f}s/张")
                break
            
            # 处理空输入
            if not input_path:
                print("❌ 请输入有效的图片路径")
                continue
            
            # 处理引号
            input_path = input_path.strip('"\'')
            
            # 处理图像
            start_time = time.time()
            result = service.process_image(input_path, compression_quality)
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
