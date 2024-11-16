"""
This script processes COLMAP output files (`images.txt`, `points3D.txt`, `cameras.txt`) and generates a `poses_bounds.npy` file. 
The generated file contains camera poses and depth bounds, which are typically used in 3D reconstruction pipelines like NeRF.

### Workflow:

1. **Parsing Input Files**:
   - `parse_images(images_file)`: Parses `images.txt` to extract camera poses, 2D points, and image metadata.
   - `parse_points3D(points3D_file)`: Parses `points3D.txt` to extract 3D points and their associated properties.
   - `parse_cameras(cameras_file)`: Parses `cameras.txt` to extract camera intrinsic parameters.

2. **Data Processing**:
   - Computes camera poses using quaternion-based rotation matrices.
   - Calculates near and far depth bounds based on the 3D points corresponding to the 2D image points.

3. **Saving Results**:
   - Combines camera poses and depth bounds into a single array.
   - Saves the resulting data into a `.npy` file (`poses_bounds.npy`).

### Usage:
1. Update the paths for `images_file`, `points3D_file`, and `cameras_file` to point to your COLMAP output files.
2. Set the `output_file` path to where you want to save the `poses_bounds.npy` file.
3. Run the script. The processed data will be saved in the specified location.

### Requirements:
- Python 3.x
- NumPy library. Install using:
"""

import os
import numpy as np

# 解析 images.txt 文件
def parse_images(images_file):
    images = {}
    with open(images_file, 'r') as f:
        image_id = None  # Track the current image ID for points2D assignment
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            
            # Skip comments and empty lines
            if line.startswith('#') or line == '':
                continue
            
            parts = line.split()
            
            # Check if the line corresponds to the first line of each image block
            if len(parts) == 10:
                try:
                    image_id = int(parts[0])
                    qw, qx, qy, qz = map(float, parts[1:5])
                    tx, ty, tz = map(float, parts[5:8])
                    camera_id = int(parts[8])
                    name = parts[9]
                    
                    # Store image pose and metadata
                    images[image_id] = {
                        'quaternion': [qw, qx, qy, qz],
                        'translation': [tx, ty, tz],
                        'camera_id': camera_id,
                        'name': name,
                        'points2d': []
                    }
                except ValueError as e:
                    print(f"Error parsing image metadata line: {line}, error: {e}")
            elif len(parts) % 3 == 0:
                # Parse 2D points and corresponding 3D point IDs
                try:
                    points = [(float(parts[i]), float(parts[i+1]), int(parts[i+2])) for i in range(0, len(parts), 3)]
                    if image_id is not None and image_id in images:
                        images[image_id]['points2d'].extend(points)
                except ValueError as e:
                    print(f"Error parsing points2D line: {line}, error: {e}")
            else:
                print(f"Warning: Unexpected line format in image file: {line}")
                
    return images

# 解析 points3D.txt 文件
def parse_points3D(points3D_file):
    points3D = {}
    with open(points3D_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or line == '':
                continue
            
            parts = line.split()
            try:
                point3D_id = int(parts[0])
                x, y, z = map(float, parts[1:4])
                r, g, b = map(int, parts[4:7])
                error = float(parts[7])
                track = [(int(parts[i]), int(parts[i+1])) for i in range(8, len(parts), 2)]
                
                # Store 3D point data
                points3D[point3D_id] = {
                    'xyz': [x, y, z],
                    'rgb': [r, g, b],
                    'error': error,
                    'track': track
                }
            except ValueError as e:
                print(f"Error parsing 3D point line: {line}, error: {e}")
                
    return points3D

# 解析 cameras.txt 文件
def parse_cameras(cameras_file):
    cameras = {}
    with open(cameras_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or line == '':
                continue
            
            parts = line.split()
            try:
                camera_id = int(parts[0])
                model = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                params = list(map(float, parts[4:]))
                
                # Store camera parameters
                cameras[camera_id] = {
                    'model': model,
                    'width': width,
                    'height': height,
                    'params': params
                }
            except ValueError as e:
                print(f"Error parsing camera line: {line}, error: {e}")
                
    return cameras

# 将数据存储到 poses_bounds.npy
def save_poses_bounds(images, points3D, cameras, output_file):
    poses = []
    bounds = []
    
    for image_id, image_data in images.items():
        qw, qx, qy, qz = image_data['quaternion']
        tx, ty, tz = image_data['translation']
        
        # 计算旋转矩阵
        R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
        T = np.array([tx, ty, tz])
        pose = np.concatenate([R, T.reshape(3, 1)], axis=1)
        
        # 使用相机的内参
        camera_id = image_data['camera_id']
        if camera_id in cameras:
            camera_params = cameras[camera_id]['params']
            focal_length = camera_params[0] if len(camera_params) > 0 else 1.0
            pose = np.hstack([pose, [[focal_length], [0], [0]]])
        
        # 计算 near 和 far
        near, far = compute_near_far(points3D, image_data['points2d'])
        bounds.append([near, far])
        
        poses.append(pose.ravel())
    
    # 将数据保存为 npy 文件
    poses_bounds = np.hstack([np.array(poses), np.array(bounds)])
    np.save(output_file, poses_bounds)
    print(f"Poses and bounds saved to {output_file}")

# 辅助函数：将四元数转换为旋转矩阵
def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    return R

# 辅助函数：计算 near 和 far 参数
def compute_near_far(points3D, points2D):
    depths = []
    for x, y, point3D_id in points2D:
        if point3D_id != -1 and point3D_id in points3D:
            depths.append(np.linalg.norm(points3D[point3D_id]['xyz']))
    if len(depths) == 0:
        return 1.0, 10.0  # 默认值
    return np.percentile(depths, 5), np.percentile(depths, 95)

# 使用示例
images_file = 'E:/CS5330/colmap-x64-windows-cuda/project/temple/images.txt'  # 替换为你的 images.txt 路径
points3D_file = 'E:/CS5330/colmap-x64-windows-cuda/project/temple/points3D.txt'  # 替换为你的 points3D.txt 路径
cameras_file = 'E:/CS5330/colmap-x64-windows-cuda/project/temple/cameras.txt'  # 替换为你的 cameras.txt 路径
output_file = 'E:/CS5330/colmap-x64-windows-cuda/project/temple/poses_bounds.npy'  # 保存 poses_bounds.npy 的路径

images = parse_images(images_file)
points3D = parse_points3D(points3D_file)
cameras = parse_cameras(cameras_file)
save_poses_bounds(images, points3D, cameras, output_file)
