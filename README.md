# CS5330_Final_Project

# Reducing the Impact of Irrelevant Objects in 3D Reconstruction Using NeRF

## Introduction
This project aims to enhance the quality of 3D reconstructions using NeRF by reducing the impact of irrelevant objects. We leverage Mask R-CNN for object segmentation and a diffusion model for inpainting, allowing us to create cleaner training data for improved NeRF performance.

### Key Approaches
1. **NeRF (Neural Radiance Fields):** A state-of-the-art approach for generating high-quality 3D models from 2D images.
2. **Segmentation Models:**
   - **Mask R-CNN:** Used for instance segmentation to detect and mask irrelevant objects within the images.
3. **Diffusion Model Inpainting:** Fills in masked areas with realistic details, enhancing the visual consistency of the training images and ensuring smoother NeRF outputs.
