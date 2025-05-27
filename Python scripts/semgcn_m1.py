import sys
import os
import json
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

# Update sys.path to include the mmpose directories
sys.path.append(r"C:/Users/MYSEL/Desktop/mmpose-updated/mmpose")
sys.path.append(r"C:/Users/MYSEL/Desktop/mmpose-updated/mmpose/models")

from sem_gcn import SemGCN

# Set model parameters (must match checkpoint's configuration)
num_joints = 17  # Using Human3.6M 17 joints
hid_dim = 128         # hidden dimension; must match training config
coords_dim = (2, 3)   # lifting from 2D to 3D
num_layers = 4
nodes_group = None
p_dropout = 0.25

# Build an adjacency matrix for 17 joints (simple chain structure for initialization)
adjacency = np.zeros((num_joints, num_joints), dtype=np.float32)
for i in range(num_joints - 1):
    adjacency[i, i + 1] = 1
    adjacency[i + 1, i] = 1
adjacency = torch.tensor(adjacency)

# Initialize the SemGCN model
model = SemGCN(adj=adjacency, hid_dim=hid_dim, coords_dim=coords_dim,
               num_layers=num_layers, nodes_group=nodes_group, p_dropout=p_dropout)

# Load model checkpoint
checkpoint_path = r"C:/Users/MYSEL/Desktop/mmpose-updated/mmpose/ckpt_semgcn.pth"
checkpoint = torch.load(checkpoint_path, map_location='cpu')
state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

# Adapt checkpoint weights
model_state = model.state_dict()
for key in state_dict:
    if key in model_state:
        ckpt_tensor = state_dict[key]
        model_tensor = model_state[key]
        if ckpt_tensor.shape != model_tensor.shape:
            if len(ckpt_tensor.shape) == 2 and ckpt_tensor.shape[1] > model_tensor.shape[1]:
                state_dict[key] = ckpt_tensor[:, :model_tensor.shape[1]]
                print(f"Adapted {key}: sliced from {ckpt_tensor.shape} to {state_dict[key].shape}")
            else:
                print(f"Warning: {key} shape mismatch cannot be adapted automatically \
                      (checkpoint: {ckpt_tensor.shape}, expected: {model_tensor.shape}).")
model.load_state_dict(state_dict)
model.eval()

# Load 2D keypoints
input_json_path = r"C:/Users/MYSEL/Desktop/mmpose-updated/mmpose/vis_results/dataset/results_fjump.json"
with open(input_json_path, 'r') as f:
    data = json.load(f)
frames_info = data.get("instance_info", [])
if not frames_info:
    raise ValueError("No 'instance_info' found in the input JSON file.")

keypoints_list = []
frame_ids = []
for frame in frames_info:
    frame_id = frame.get("frame_id", None)
    instances = frame.get("instances", [])
    if not instances:
        continue
    instance = instances[0]
    kp = instance.get("keypoints", [])
    if len(kp) != 17:
        raise ValueError(f"Expected 17 keypoints in frame {frame_id}, but got {len(kp)}.")
    keypoints_list.append(kp)
    frame_ids.append(frame_id)

keypoints_np = np.array(keypoints_list, dtype=np.float32)
print("Original 2D keypoints shape:", keypoints_np.shape)

input_tensor = torch.tensor(keypoints_np)
with torch.no_grad():
    output_3d = model(input_tensor)
output_3d_np = output_3d.cpu().numpy()

# Save 3D pose
time_series_output = {"all_pose_3d": []}
for idx, keypoints_3d in enumerate(output_3d_np):
    time_series_output["all_pose_3d"].append({
        "frame": frame_ids[idx],
        "keypoints": keypoints_3d.tolist()
    })
output_json_path = r"C:/Users/MYSEL/Desktop/mmpose-updated/mmpose/vis_results/dataset_3d/3d/fjump_result_time_series_3d.json"
with open(output_json_path, 'w') as f:
    json.dump(time_series_output, f, indent=4)
print("Time-series 3D keypoints saved to", output_json_path)

# Compute kinematics
num_frames = output_3d_np.shape[0]
velocities = np.zeros_like(output_3d_np)
for i in range(num_frames - 1):
    velocities[i] = output_3d_np[i + 1] - output_3d_np[i]
velocities[-1] = velocities[-2]

accelerations = np.zeros_like(velocities)
for i in range(num_frames - 1):
    accelerations[i] = velocities[i + 1] - velocities[i]
accelerations[-1] = accelerations[-2]

def compute_angle(v1, v2):
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return None
    cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

neighbors = {
    1: [0, 2], 2: [1, 3], 4: [0, 5], 5: [4, 6], 9: [8, 10],
    11: [8, 12], 12: [11, 13], 14: [8, 15], 15: [14, 16]
}

joint_angles_all = []
for frame in output_3d_np:
    angles_frame = {}
    for joint, nbrs in neighbors.items():
        vec1 = frame[nbrs[0]] - frame[joint]
        vec2 = frame[nbrs[1]] - frame[joint]
        angle = compute_angle(vec1, vec2)
        angles_frame[str(joint)] = angle if angle is not None else None
    joint_angles_all.append(angles_frame)

kinematic_output = {"frames": []}
for idx in range(num_frames):
    kinematic_output["frames"].append({
        "frame": frame_ids[idx],
        "joint_velocities": velocities[idx].tolist(),
        "joint_accelerations": accelerations[idx].tolist(),
        "joint_angles": joint_angles_all[idx]
    })
kinematic_json_path = r"C:/Users/MYSEL/Desktop/mmpose-updated/mmpose/vis_results/dataset_3d/3d/fjump_result_kinematics.json"
with open(kinematic_json_path, 'w') as f:
    json.dump(kinematic_output, f, indent=4)
print("Kinematic metrics saved to", kinematic_json_path)

# Visualization and video
skeleton = [
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16)
]

video_path = r"C:/Users/MYSEL/Desktop/mmpose-updated/mmpose/vis_results/dataset_3d/3d/video_fjump.mp4"
frame_width, frame_height = 640, 480
fps = 10
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))
plt.ioff()

# Automatically divide view labels based on total number of frames
third = num_frames // 3

for frame_idx, keypoints_3d in enumerate(output_3d_np):
    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(111, projection='3d')

    x_vals = keypoints_3d[:, 0]
    y_vals = keypoints_3d[:, 1]
    z_vals = keypoints_3d[:, 2]

    for idx, (x, y, z) in enumerate(zip(x_vals, y_vals, z_vals)):
        ax.scatter(x, y, z, c='b', marker='o')
        ax.text(x, y, z, str(idx), color='black', fontsize=8)

    for connection in skeleton:
        i, j = connection
        if i < num_joints and j < num_joints:
            ax.plot(
                [keypoints_3d[i, 0], keypoints_3d[j, 0]],
                [keypoints_3d[i, 1], keypoints_3d[j, 1]],
                [keypoints_3d[i, 2], keypoints_3d[j, 2]], c='r', linewidth=2)

    ax.set_xlim(np.min(x_vals)-10, np.max(x_vals)+10)
    ax.set_ylim(np.min(y_vals)-10, np.max(y_vals)+10)
    ax.set_zlim(np.min(z_vals)-10, np.max(z_vals)+10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_title("3D Pose Estimation", fontsize=10)
    ax.text2D(0.02, 0.95, f"Frame: {frame_idx}", transform=ax.transAxes, fontsize=10, color='blue')

    if frame_idx < third:
        view_label = "Front View"
    elif frame_idx < 2 * third:
        view_label = "Left View"
    else:
        view_label = "Right View"
    ax.text2D(0.75, 0.95, view_label, transform=ax.transAxes, fontsize=12, color='green', weight='bold')

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_bgr = cv2.resize(img_bgr, (frame_width, frame_height))

    video_writer.write(img_bgr)
    plt.close(fig)

video_writer.release()
print("Output video saved to", video_path)
