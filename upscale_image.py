import os
import time
import cv2
import numpy as np
import torch
from PIL import Image
import argparse
from datetime import datetime

class RealESRGANer:
    """Real-ESRGAN inference class"""
    
    def __init__(self, model_path, device='cpu', scale=4):
        self.device = device
        self.scale = scale
        self.model_path = model_path
        
        # Load model
        self.model = self.load_model()
        
    def load_model(self):
        """Load the Real-ESRGAN model"""
        try:
            # Import RRDBNet from basicsr
            from basicsr.archs.rrdbnet_arch import RRDBNet
            
            # Initialize model architecture
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                          num_block=6, num_grow_ch=32, scale=4)
            
            # Load pretrained weights
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if 'params_ema' in checkpoint:
                model.load_state_dict(checkpoint['params_ema'])
            elif 'params' in checkpoint:
                model.load_state_dict(checkpoint['params'])
            else:
                model.load_state_dict(checkpoint)
                
            model.eval()
            model = model.to(self.device)
            
            return model
            
        except ImportError:
            print("BasicSR not found. Installing required dependencies...")
            os.system("pip install basicsr")
            from basicsr.archs.rrdbnet_arch import RRDBNet
            
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                          num_block=6, num_grow_ch=32, scale=4)
            
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if 'params_ema' in checkpoint:
                model.load_state_dict(checkpoint['params_ema'])
            elif 'params' in checkpoint:
                model.load_state_dict(checkpoint['params'])
            else:
                model.load_state_dict(checkpoint)
                
            model.eval()
            model = model.to(self.device)
            
            return model
    
    def enhance(self, img):
        """Enhance image using Real-ESRGAN"""
        # Convert to tensor
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        img = img.unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(img)
        
        # Convert back to numpy
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output, (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        
        return output

def resize_image_keep_aspect_ratio(image, max_size=512):
    """Resize image keeping aspect ratio with max dimension of max_size"""
    h, w = image.shape[:2]
    
    # Calculate scaling factor
    scale_factor = min(max_size / w, max_size / h)
    
    # Only resize if image is larger than max_size
    if scale_factor < 1:
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    return image, scale_factor

def log_message(message, log_file="log.txt"):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}\n"
    print(message)
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_entry)

def main():
    # Configuration
    input_file = "TEST.png"
    model_path = "RealESRGAN_x4plus_anime_6B.pth"
    max_resize = 512
    device = "cpu"
    
    # Initialize log
    log_file = "log.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Real-ESRGAN Image Upscaling Log\n")
        f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n")
    
    start_time = time.time()
    
    try:
        # Check if input file exists
        if not os.path.exists(input_file):
            log_message(f"Error: Input file '{input_file}' not found!")
            return
        
        if not os.path.exists(model_path):
            log_message(f"Error: Model file '{model_path}' not found!")
            return
        
        log_message(f"Starting image upscaling process...")
        log_message(f"Input file: {input_file}")
        log_message(f"Model: {model_path}")
        log_message(f"Device: {device}")
        log_message(f"Max resize dimension: {max_resize}")
        
        # Load input image
        log_message("Loading input image...")
        img = cv2.imread(input_file, cv2.IMREAD_COLOR)
        if img is None:
            log_message(f"Error: Could not load image '{input_file}'")
            return
        
        original_shape = img.shape
        log_message(f"Original image size: {original_shape[1]}x{original_shape[0]}")
        
        # Resize image if needed
        log_message("Resizing image to fit max dimension...")
        img_resized, scale_factor = resize_image_keep_aspect_ratio(img, max_resize)
        resized_shape = img_resized.shape
        log_message(f"Resized image size: {resized_shape[1]}x{resized_shape[0]} (scale factor: {scale_factor:.3f})")
        
        # Initialize Real-ESRGAN
        log_message("Initializing Real-ESRGAN model...")
        model_start_time = time.time()
        upsampler = RealESRGANer(model_path=model_path, device=device, scale=4)
        model_load_time = time.time() - model_start_time
        log_message(f"Model loaded in {model_load_time:.2f} seconds")
        
        # Perform upscaling
        log_message("Starting image upscaling...")
        upscale_start_time = time.time()
        output_img = upsampler.enhance(img_resized)
        upscale_time = time.time() - upscale_start_time
        log_message(f"Upscaling completed in {upscale_time:.2f} seconds")
        
        output_shape = output_img.shape
        log_message(f"Output image size: {output_shape[1]}x{output_shape[0]}")
        
        # Generate output filename
        base_name = os.path.splitext(input_file)[0]
        output_file = f"ESR-{base_name}.webp"
        
        # Convert BGR to RGB for PIL
        output_img_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
        
        # Save as WebP
        log_message(f"Saving output image as {output_file}...")
        save_start_time = time.time()
        pil_img = Image.fromarray(output_img_rgb)
        pil_img.save(output_file, "WEBP", quality=95, method=6)
        save_time = time.time() - save_start_time
        log_message(f"Image saved in {save_time:.2f} seconds")
        
        # Calculate total time
        total_time = time.time() - start_time
        log_message(f"Total processing time: {total_time:.2f} seconds")
        log_message("Process completed successfully!")
        
        # Summary
        log_message("\n" + "="*50)
        log_message("SUMMARY:")
        log_message(f"Input: {input_file} ({original_shape[1]}x{original_shape[0]})")
        log_message(f"Resized: {resized_shape[1]}x{resized_shape[0]}")
        log_message(f"Output: {output_file} ({output_shape[1]}x{output_shape[0]})")
        log_message(f"Model loading time: {model_load_time:.2f}s")
        log_message(f"Upscaling time: {upscale_time:.2f}s")
        log_message(f"Saving time: {save_time:.2f}s")
        log_message(f"Total time: {total_time:.2f}s")
        
    except Exception as e:
        error_time = time.time() - start_time
        log_message(f"Error occurred after {error_time:.2f} seconds: {str(e)}")
        import traceback
        log_message(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
