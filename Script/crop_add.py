import os
from PIL import Image

input_folder = 'E:/CS5330/datasets/temple_nara_japan/temple_nara_japan/dense_crop/images'
output_folder = 'E:/CS5330/datasets/temple_nara_japan/temple_nara_japan/dense_crop/images_cropped_6_9_base'
addition_folder = 'E:/CS5330/datasets/temple_nara_japan/temple_nara_japan/dense_crop/images_cropped_6_9_base_additional'

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
