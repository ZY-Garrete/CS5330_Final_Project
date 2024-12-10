"""
This script processes images in the input folder by either cropping or resizing them to meet the target resolution
of 900x600 pixels. The script performs the following operations:

1. **Cropping Images**:
   - For images with dimensions greater than or equal to 900x600, the script crops the image to 900x600 pixels,
     keeping the content centered.

2. **Resizing Images**:
   - For images with dimensions smaller than 900x600, the script:
     a. Proportionally crops the image to maintain a 3:2 aspect ratio, centering the content.
     b. Resizes the cropped image to 900x600 pixels.

3. **File Organization**:
   - Cropped images are saved in the `output_folder`.
   - Resized images are saved in the `addition_folder`.

### Usage:
- Modify the `input_folder` variable to specify the directory containing the input images.
- Modify the `output_folder` and `addition_folder` variables to set the output directories for cropped and resized images.
- Ensure that the script has permission to read from the input folder and write to the output folders.

### Supported Image Formats:
- The script processes `.jpg`, `.jpeg`, and `.png` image files.

### Requirements:
- Python 3.x
- The `Pillow` library (PIL fork) for image processing. Install using:
"""

import os
from PIL import Image

input_folder = 'E:/CS5330/colmap-x64-windows-cuda/colmap_12_5/images_sorted'
output_folder = 'E:/CS5330/colmap-x64-windows-cuda/colmap_12_5/images_crop/images_cropped_6_9_base'
addition_folder = 'E:/CS5330/colmap-x64-windows-cuda/colmap_12_5/images_crop/images_cropped_6_9_base_additional'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)
os.makedirs(addition_folder, exist_ok=True)

# 目标分辨率
target_width = 900
target_height = 600

# 遍历文件夹中的所有文件
for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)

    # 检查文件是否为图片格式（例如jpg、jpeg、png）
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        with Image.open(file_path) as img:
            width, height = img.size

            # 对于分辨率大于900x600的图片，直接居中裁剪到900x600
            if width >= target_width and height >= target_height:
                left = (width - target_width) / 2
                top = (height - target_height) / 2
                right = left + target_width
                bottom = top + target_height

                cropped_img = img.crop((left, top, right, bottom))
                cropped_img.save(os.path.join(output_folder, filename))
                print(f"Cropped and saved {filename} to {output_folder}")

            # 对于分辨率小于900x600的图片，先等比例裁剪再resize放大
            elif width < target_width or height < target_height:
                # 保持原始宽度，将高度裁剪为宽度的三分之二
                new_height = width * 2 // 3

                # 确保裁剪的高度不超过原始高度，居中裁剪
                if new_height <= height:
                    top = (height - new_height) / 2
                    bottom = top + new_height
                    temp_cropped_img = img.crop((0, top, width, bottom))

                    # 将裁剪后的图片resize到900x600
                    resized_img = temp_cropped_img.resize((target_width, target_height), Image.LANCZOS)
                    resized_img.save(os.path.join(addition_folder, filename))
                    print(f"Resized and saved {filename} to {addition_folder}")
                else:
                    print(f"Skipping {filename}: unable to process due to unexpected dimensions ({width}x{height})")

print("Processing completed.")
