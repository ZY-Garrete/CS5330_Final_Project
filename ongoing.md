
# Research Plan in Nov
+ **1. Datasets**
  + Analysis Photourism datdaset: ensure that the images contain non-relevant objects, and prepare them for use by Mask R-CNN.
  + Collect and preprocess possible test datasets for later NeRF reconstruction testing.
  + Colmap-sparse matrix.bin 

+ **2. Complete the reproduction of R-CNN code**
  + Idea adjustment(mask RCNN -> mask2former)
  + Training Fast model, performing object detection task on prepared dataset, and generating masks.

+ **3. Complete the reproduction of DDPM code**
  + Idea adjustment(DDPM)
  + Use diffusion model to perform repair test on non-related object areas, and try to fill reasonable content in the mask area.
  
+ **4. intergrate Mask2Former + DDOM**
  + Cooperation to integrate Fast R-CNN and diffusion model to complete the "detection-separation-repair" process and ensure that the output image is suitable for 3D reconstruction

+ **5. Run NeRF on the integrated image for 3D reconstruction**
  + compare the effect difference with and without non-related object repair.

+ **Fine tune and documentation**

# Ongoing process
+ Get the available generation results 
+ Colmap reconstruction to get the sparse bin file. (convert it to npy file for nerf)
+ padding script

# Finished process
+  IMC PhotoTourism datasets(Image Matching Challenge Phototourism)
   +  preprocess image(11.14) Uniform size 
   +  refilter image based the quality(11.13)
+ Implement the batch process of image sets by Mask_DDPM
  + Normalize the label map. （11.14）
  + Solve the combined feature mask segment.（11.14） 
  + optimize the prompt( ... )
+ **Complete the reproduction of NeRF code**
    + prepare the available image set and its sparse file
+ **script helper**
  + images_crop
  + image_resize
  + pose_pound 
# Questions and ideas
  + ~~Relevance of the image generated by the diffusion model~~
  + ~~size difference~~ 
  + **About the output and data flow of nerf**
  + ~~the data flow transfer difference in tensorflow and pytorch~~
