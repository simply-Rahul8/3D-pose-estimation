import numpy as np
import json
from filterpy.kalman import KalmanFilter

# Load the 2D and 3D keypoints data
def load_keypoints():
    # Load the 2D and 3D keypoints from the correct files
    with open('C:/Users/MYSEL/Desktop/mmpose-updated/mmpose/vis_results/dataset/results_fwalk.json', 'r', encoding='utf-8') as f2d, \
         open('C:/Users/MYSEL/Desktop/mmpose-updated/mmpose/vis_results/dataset_3d/3d/fwalk_result_time_series_3d.json', 'r', encoding='utf-8') as f3d:
        data_2d = json.load(f2d)  # 2D keypoints
        data_3d = json.load(f3d)  # 3D keypoints
    
    return data_2d, data_3d

# Function to calculate the bounding box of 2D keypoints
def calculate_2d_bounding_box(keypoints_2d):
    keypoints_2d = np.array(keypoints_2d)
    x_min, y_min = np.min(keypoints_2d, axis=0)
    x_max, y_max = np.max(keypoints_2d, axis=0)
    return x_max - x_min, y_max - y_min  # Return the width and height of the bounding box

# Normalize 3D keypoints using the average pairwise distance between keypoints
def normalize_3d_keypoints_by_pairwise_distance(keypoints_3d):
    keypoints_3d = np.array(keypoints_3d)
    
    # Calculate the pairwise distances between all keypoints
    distances = np.linalg.norm(keypoints_3d[:, np.newaxis] - keypoints_3d, axis=2)
    
    # Calculate the average distance between all pairs of keypoints
    avg_distance = np.mean(distances[distances != 0])  # Exclude the diagonal (self-to-self distances)
    
    # Normalize the 3D keypoints by the average distance
    normalized_3d = keypoints_3d / avg_distance
    return normalized_3d

# Function to smooth 2D keypoints using a Kalman filter
def kalman_filter_smoothing(keypoints_2d):
    smoothed_keypoints = []
    
    # Initialize Kalman Filter for 2D tracking (x, y positions)
    kf = KalmanFilter(dim_x=2, dim_z=2)
    kf.P *= 1000.  # Large initial uncertainty
    kf.R = np.eye(2)  # Measurement noise (assume perfect measurements)
    kf.Q = np.eye(2)  # Process noise
    
    for keypoint in keypoints_2d:
        kf.predict()
        kf.update(keypoint)
        smoothed_keypoints.append(kf.x)
    
    return np.array(smoothed_keypoints)

# Function to calculate Weighted MPJPE
def calculate_weighted_mpjpe(data_2d, data_3d):
    # Ensure that both data have the same number of frames
    num_frames = len(data_2d['instance_info'])
    assert num_frames == len(data_3d['all_pose_3d']), "Number of frames in 2D and 3D data do not match."
    
    mpjpe_values = []
    
    # Iterate through each frame
    for frame_idx in range(num_frames):
        # Extract keypoints for the current frame (2D and 3D)
        keypoints_2d = data_2d['instance_info'][frame_idx]['instances'][0]['keypoints']  # 2D
        keypoints_3d = data_3d['all_pose_3d'][frame_idx]['keypoints']  # 3D
        confidence_scores = data_2d['instance_info'][frame_idx]['instances'][0]['keypoint_scores']
        
        # Smooth the 2D keypoints using Kalman filter
        keypoints_2d_smooth = kalman_filter_smoothing(keypoints_2d)
        
        # Calculate the 2D bounding box size
        bbox_width, bbox_height = calculate_2d_bounding_box(keypoints_2d)
        
        # Normalize the 3D keypoints based on the pairwise distance
        keypoints_3d_normalized = normalize_3d_keypoints_by_pairwise_distance(keypoints_3d)
        
        # Project 3D keypoints onto the 2D plane by ignoring the z-coordinate
        keypoints_3d_2d = keypoints_3d_normalized[:, :2]  # Ignore z coordinate
        
        # Ensure both visible_2d and visible_3d_2d have matching shapes
        visible_2d = np.array([kp[:2] for kp in keypoints_2d_smooth])  # Only (x, y) for visible 2D keypoints
        visible_3d_2d = np.array(keypoints_3d_2d)  # Already projected 3D to 2D
        
        # Flatten the 3D to 2D keypoints to remove any extra dimension (make sure it's 17x2)
        visible_2d = visible_2d.reshape(-1, 2)
        visible_3d_2d = visible_3d_2d.reshape(-1, 2)
        
        # Debugging the final shapes before calculating MPJPE
        print(f"Shape of visible_2d: {visible_2d.shape}")
        print(f"Shape of visible_3d_2d: {visible_3d_2d.shape}")
        
        # Check if the shapes match
        if visible_2d.shape != visible_3d_2d.shape:
            print(f"Shape mismatch detected: {visible_2d.shape} vs {visible_3d_2d.shape}")
            continue
        
        # Calculate the weighted MPJPE
        distances = np.linalg.norm(visible_3d_2d - visible_2d, axis=1)  # Compare x, y only
        weighted_distances = distances * confidence_scores  # Weight by confidence score
        frame_mpjpe = np.sum(weighted_distances) / np.sum(confidence_scores)  # Weighted average
        
        mpjpe_values.append(frame_mpjpe)
    
    # Final MPJPE: mean over all frames
    overall_mpjpe = np.mean(mpjpe_values)
    
    return overall_mpjpe

# Main execution to load the data and calculate Weighted MPJPE
def main():
    data_2d, data_3d = load_keypoints()  # Load the 2D and 3D keypoints
    mpjpe_result = calculate_weighted_mpjpe(data_2d, data_3d)  # Calculate Weighted MPJPE
    print(f"Weighted Mean Per Joint Position Error (MPJPE): {mpjpe_result}")

if __name__ == "__main__":
    main()
