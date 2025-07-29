# Real-ESRGAN Image Upscaling Script

This script uses the RealESRGAN_x4plus_anime_6B model to upscale images with the following features:

## Features
- Automatically resizes input images to have a maximum dimension of 512 pixels before upscaling
- Uses CPU for processing (no GPU required)
- Outputs upscaled images in WebP format with "ESR-" prefix
- Logs detailed timing information to `log.txt`
- 4x upscaling using the anime-optimized model

## Requirements
- Python 3.7+
- Dependencies listed in `requirements.txt`

## Installation
```bash
pip install -r requirements.txt
```

## Usage
1. Place your input image as `TEST.png` in the same directory
2. Ensure `RealESRGAN_x4plus_anime_6B.pth` model file is present
3. Run the script:
```bash
python upscale_image.py
```

## Output
- Upscaled image: `ESR-TEST.webp`
- Processing log: `log.txt`

## Processing Flow
1. Load input image (`TEST.png`)
2. Resize image proportionally so the longest side is 512 pixels
3. Load RealESRGAN model
4. Perform 4x upscaling
5. Save result as WebP format
6. Log all timing information

## Example Results
- Input: `TEST.png` (144x148)
- After resize: 144x148 (no resize needed as already under 512)
- Output: `ESR-TEST.webp` (576x592) - 4x upscaled
- Total processing time: ~5.5 seconds on CPU

## Model Information
The script uses `RealESRGAN_x4plus_anime_6B.pth`, which is specifically optimized for anime/cartoon images and provides excellent results for this type of content.
