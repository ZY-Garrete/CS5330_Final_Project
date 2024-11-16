import os
from tqdm import tqdm
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cityscapes class label
CITYSCAPES_CLASS_NAMES = [
    "unlabeled", "ego vehicle", "rectification border", "out of roi", "static", "dynamic", "ground", "road",
    "sidewalk", "parking", "rail track", "building", "wall", "fence", "guard rail", "bridge", "tunnel", 
    "pole", "polegroup", "traffic light", "traffic sign", "vegetation", "terrain", "sky", "person", 
    "rider", "car", "truck", "bus", "caravan", "trailer", "train", "motorcycle", "bicycle", "license plate"
]

# load Mask2Former 
image_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-cityscapes-panoptic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-cityscapes-panoptic").to(device)

# load Stable Diffusion inpainting
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")  


def process_image(image_path, output_visible_dir, output_inpainted_dir, prompt):
    #preprocess
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    inputs = image_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # 语义分割图
    pred_instance_map = image_processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0]
    segmentation_map = pred_instance_map.cpu().numpy()

    # 获取 "sidewalk(Tree)" 和 "building(person)" 类别的索引
    sidewalk_class = CITYSCAPES_CLASS_NAMES.index("sidewalk")
    building_class = CITYSCAPES_CLASS_NAMES.index("building")

    # combine Tree and person as final mask
    combined_mask = (segmentation_map == sidewalk_class) | (segmentation_map == building_class)
    mask_image = (combined_mask.astype(np.uint8) * 255)  # 转换为二值图像 (0 或 255)

    # image without tree and person
    only_areas_image = np.zeros_like(image_np)
    only_areas_image[~combined_mask] = image_np[~combined_mask]  # 保留非遮盖区域
    only_areas_pil = Image.fromarray(only_areas_image)

    # 转换为 PIL 图像格式
    mask_pil = Image.fromarray(mask_image).convert("RGB")

    # inpainting
    result = pipe(prompt=prompt, image=image, mask_image=mask_pil).images[0]

    filename = os.path.basename(image_path)
    only_areas_pil.save(os.path.join(output_visible_dir, filename))
    result.save(os.path.join(output_inpainted_dir, filename))

def process_dataset(input_dir, output_visible_dir, output_inpainted_dir, prompt):
    # output file
    os.makedirs(output_visible_dir, exist_ok=True)
    os.makedirs(output_inpainted_dir, exist_ok=True)

    # get all image
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    # tqdm
    for filename in tqdm(image_files, desc="Processing Images"):
        image_path = os.path.join(input_dir, filename)
        process_image(image_path, output_visible_dir, output_inpainted_dir, prompt)

    print("Processing complete!")

if __name__ == "__main__":
    # 设置路径
    input_dir = "E:/CS5330/datasets/temple_nara_japan/temple_nara_japan/dense_crop/image_6_9_all/All_1"  # 输入图片文件夹路径
    output_visible_dir = "E:/CS5330/datasets/temple_nara_japan/temple_nara_japan/dense_crop/image_6_9_all/Person and Tree Masked Inpaint/only_areas_visible"  # 输出Only Areas Visible图片路径
    output_inpainted_dir = "E:/CS5330/datasets/temple_nara_japan/temple_nara_japan/dense_crop/image_6_9_all/Person and Tree Masked Inpaint/inpainted_images"  # 输出Inpainted图片路径
    prompt = "A peaceful Japanese temple scene with clean stone pathways and traditional wooden temple walls,"\
    "no tourists, no trees or plants blocking the view, no extra temples, photorealistic and true to Todaiji Temple's historic architecture." # 修复的提示词

    # 批量处理数据集
    process_dataset(input_dir, output_visible_dir, output_inpainted_dir, prompt)
