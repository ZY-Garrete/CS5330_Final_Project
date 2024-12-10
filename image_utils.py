from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


def load_segment_model():
    # 提取预训练标签
    CITYSCAPES_CLASS_NAMES = [
        "unlabeled", "ego vehicle", "rectification border", "out of roi", "static", "dynamic", "ground", "road",
        "sidewalk", "parking", "rail track", "building", "wall", "fence", "guard rail", "bridge", "tunnel", 
        "pole", "polegroup", "traffic light", "traffic sign", "vegetation", "terrain", "sky", "person", 
        "rider", "car", "truck", "bus", "caravan", "trailer", "train", "motorcycle", "bicycle", "license plate"
    ]

    # 提取模型
    image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-cityscapes-panoptic")
    model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-cityscapes-panoptic")

    return (CITYSCAPES_CLASS_NAMES, image_processor, model)

def load_difussion_model():
    # 加载pipline
    # more details on this model: https://huggingface.co/docs/diffusers/main/en/using-diffusers/inpaint
    pipe = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")  # Use GPU if available
    return pipe

def square_padding(image, mask=None):
    # 转换图像为 NumPy 格式
    image_np = np.array(image)
    H, W = image_np.shape[:2]
    canvas_size = max(H, W)

    # 初始化画布和遮罩
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=image_np.dtype)
    padded_mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8) if mask is not None else None

    # 计算偏移量并填充
    top, left = (canvas_size - H) // 2, (canvas_size - W) // 2
    canvas[top:top+H, left:left+W, :] = image_np

    if mask is not None:
        mask_np = np.array(mask)
        padded_mask[top:top+H, left:left+W] = mask_np

    return Image.fromarray(canvas), padded_mask if mask is not None else None

def size_recover(ori_image, padded_image):
    # Ensure both inputs are in NumPy array form for easy manipulation
    ori_np = np.array(ori_image)
    padded_np = np.array(padded_image)

    # Original dimensions
    original_height, original_width = ori_np.shape[:2]

    # Padded dimensions should be a square of size max(original_height, original_width)
    padded_size = max(original_height, original_width)

    # Determine the current size of the padded image
    padded_current_height, padded_current_width = padded_np.shape[:2]

    # Calculate the scale factor based on how the padded dimensions relate to the largest original dimension
    scale_factor = padded_size / float(padded_current_height)  # assumes padded_current_height == padded_current_width

    # Calculate the target dimensions to scale back to original padded dimensions
    target_height = int(scale_factor * padded_current_height)
    target_width = int(scale_factor * padded_current_width)

    # Resize the image back to the size it was when it was padded
    resized_image = Image.fromarray(padded_np).resize((target_width, target_height), Image.Resampling.LANCZOS)

    # Calculate the crop to get back to the original dimensions
    start_x = (target_width - original_width) // 2
    start_y = (target_height - original_height) // 2

    # Perform the cropping to get exactly the original size
    cropped_image = resized_image.crop((start_x, start_y, start_x + original_width, start_y + original_height))

    return cropped_image

def enforce_nerf_inputs(image, filename):
    width, height = image.size
    if width >= 900 and height >= 600:
        left = (width - 900) / 2
        top = (height - 600) / 2
        cropped_img = image.crop((left, top, left + 900, top + 600))
        return cropped_img
    elif width < 900 or height < 600:
        new_height = width * 2 // 3
        if new_height <= height:
            top = (height - new_height) / 2
            temp_cropped_img = image.crop((0, top, width, top + new_height))
            resized_img = temp_cropped_img.resize((900, 600), Image.Resampling.LANCZOS)
            return resized_img
        else:
            print(f"Skipping {filename}: unable to process due to unexpected dimensions ({width}x{height})")
            return None
        
def generate_masked_image(image, mask):
    # 确保遮罩为 0-1 的范围
    if mask.max() > 1:
        mask = mask / 255.0

    # 创建绘图
    fig, ax = plt.subplots(figsize=(image.shape[1] / 100, image.shape[0] / 100), dpi=100)
    ax.imshow(image)  # 显示原始图像
    ax.imshow(mask, cmap="jet", alpha=0.5)  # 使用透明度叠加遮罩
    ax.axis("off")  # 关闭坐标轴

    # 保留原始比例
    ax.set_aspect('auto')  # 保持图像原始比例
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 去除多余边距

    # 将绘图转换为 NumPy 数组
    fig.canvas.draw()
    masked_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    masked_image = masked_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)  # 关闭绘图，释放内存
    return masked_image

def segment_img(image, segment_model):
    # 加载模型

    class_names, image_processor, model = segment_model

    image_np = np.array(image)  # 转换图片为numpy矩阵

    # 处理图片
    inputs = image_processor(images=image, return_tensors="pt")

    # 分割
    with torch.no_grad():
        outputs = model(**inputs)

    # 提取遮罩和遮罩的分数
    class_scores = outputs.class_queries_logits[0]  # [num_queries, num_classes]
    masks = outputs.masks_queries_logits[0]  # [num_queries, height, width]

    # 提取建筑物的index
    building_idx = class_names.index("building")

    # 初始化建筑遮罩
    height, width = masks.shape[1], masks.shape[2]
    building_mask = np.zeros((height, width))

    # 设置阈值
    confidence_threshold = 0.9

    # 遍历所有种类的遮罩
    for i in range(masks.shape[0]):

        class_idx = class_scores[i].argmax().item()
        confidence = torch.softmax(class_scores[i], dim=0).max().item()
        
        if confidence > confidence_threshold and class_idx == building_idx:
            mask = torch.sigmoid(masks[i]).cpu().numpy()
            building_mask = np.maximum(building_mask, mask)  # Combine all building masks

    # 匹配原尺寸
    building_mask_resized = cv2.resize(building_mask, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_LINEAR)

    # 创建遮罩
    building_mask_binary = (building_mask_resized > 0.5).astype(np.uint8)  # 1 where building, 0 elsewhere

    result_image = np.zeros_like(image_np)  # Start with a completely black image

    # Copy only the building areas from the original image to the black background
    result_image[building_mask_binary == 0] = image_np[building_mask_binary == 0]  # Keep only building areas

    return result_image, building_mask_binary

def inpaint_img(image, mask, difussion_model):

    # 转换为 PIL 图像（如果模型需要 PIL 图像格式）, recover the pixel values
    mask = (mask * 255).astype(np.uint8)

    image_pil = image if isinstance(image, Image.Image) else Image.fromarray(image.astype(np.uint8))
    mask_pil = Image.fromarray(mask)

    # 提示文本
    prompt = "Remove people and irrelevant objects from the image while preserving the architectural integrity and surrounding textures of the Brandenburger Tor. Ensure seamless blending with adjacent areas, maintaining natural transitions and consistent visual coherence. Avoid introducing new elements or altering the original composition."

    # 修复
    inpainted = difussion_model(prompt=prompt, image=image_pil, mask_image=mask_pil).images[0]
    return inpainted

def segment_and_inpaint(input_folder, seg_folder, mask_folder, inpainted_folder, segment_model, difussion_model, enforce_ratio=True):

    for file in tqdm(os.listdir(input_folder)):
        input_path = os.path.join(input_folder, file)
        if os.path.isfile(input_path):
            
            image = Image.open(input_path)

            # Segment
            segmented_image, mask = segment_img(image, segment_model)
            if enforce_ratio:
                pre_inpaint, mask = square_padding(segmented_image, mask)  # 填充图像和遮罩
            else:
                pre_inpaint = segmented_image

            # Save masks
            mask_path = os.path.join(mask_folder, f"{os.path.splitext(file)[0]}.npy")  # Replace extension with .npy
            np.save(mask_path, mask)

            # Save segmented imgs
            # masked_image = generate_masked_image(np.array(pre_inpaint), mask)
            Image.fromarray(segmented_image).save(os.path.join(seg_folder, file))

            # Inpaint imgs
            inpainted = inpaint_img(pre_inpaint, mask, difussion_model)
            inpainted = size_recover(image, inpainted)
            #inpainted = enforce_nerf_inputs(inpainted,file)
            if inpainted is not None:
                inpaint_path = os.path.join(inpainted_folder, file)
                inpainted.save(inpaint_path)
            else:
                print(f"Skipping save of {file}, low resolution!")

def main():
    input_folder = 'E:/CS5330/Instant-NGP-for-RTX-3000-and-4000/Instant-NGP-for-RTX-3000-and-4000/video_12_6/003_info/images'
    segmented_output_folder = 'E:/CS5330/Instant-NGP-for-RTX-3000-and-4000/Instant-NGP-for-RTX-3000-and-4000/video_12_6/003_info/images/segmented_imgs'
    mask_output_folder = 'E:/CS5330/Instant-NGP-for-RTX-3000-and-4000/Instant-NGP-for-RTX-3000-and-4000/video_12_6/003_info/images/img_masks'
    inpaint_img_folder = 'E:/CS5330/Instant-NGP-for-RTX-3000-and-4000/Instant-NGP-for-RTX-3000-and-4000/video_12_6/003_info/images/inpainted_imgs'
    
    # 检查输入文件夹是否存在
    if not os.path.exists(input_folder):
        print(f"输入文件夹 {input_folder} 不存在，请检查路径。")
        return
    
    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(segmented_output_folder):
        os.makedirs(segmented_output_folder)

    if not os.path.exists(mask_output_folder):
        os.makedirs(mask_output_folder)

    print('Loading segmentation model...')
    segment_model = load_segment_model()
    print('completed')

    print('Loading Difussion Model...')
    difussion_model = load_difussion_model()
    print('completed')
    
    print("开始处理图片...")
    segment_and_inpaint(input_folder, segmented_output_folder, mask_output_folder, inpaint_img_folder, segment_model,difussion_model, enforce_ratio=True)
    print(f"处理完成!")

if __name__ == "__main__":
    main()