import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D

# ----------------------------------------------------------------
# 1) Input JSON file with "all_pose_3d"
# ----------------------------------------------------------------
json_path = r"C:/Users/MYSEL/Desktop/mmpose-updated/mmpose/result_time_series_3d.json"

# ----------------------------------------------------------------
# 2) Load the time-series 3D data
# ----------------------------------------------------------------
with open(json_path, "r") as f:
    data = json.load(f)

frames = data["all_pose_3d"]  # A list of dicts: [{"frame": int, "keypoints": [[x,y,z], ...]}, ...]

all_keypoints = []
frame_ids = []
for frame_info in frames:
    keypoints_3d = np.array(frame_info["keypoints"], dtype=np.float32)  # shape (num_joints, 3)
    all_keypoints.append(keypoints_3d)
    frame_ids.append(frame_info["frame"])

all_keypoints = np.array(all_keypoints)  # shape: (num_frames, num_joints, 3)
num_frames = len(all_keypoints)
print("Loaded 3D keypoints shape:", all_keypoints.shape)

# ----------------------------------------------------------------
# 3) Define skeleton links (example: upper body in black, lower in red)
# ----------------------------------------------------------------
UPPER_BODY_LINKS = [
    (0, 1), (1, 2), (2, 3), 
    (3, 4), (2, 5)
]
LOWER_BODY_LINKS = [
    (6, 7), (7, 8), (8, 9),
    (6, 10), (10, 11)
]

# ----------------------------------------------------------------
# 4) Determine global axis limits
# ----------------------------------------------------------------
all_pts = all_keypoints.reshape(-1, 3)
x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
y_min, y_max = all_pts[:, 1].min(), all_pts[:, 1].max()
z_min, z_max = all_pts[:, 2].min(), all_pts[:, 2].max()

# Optional: add padding
padding = 50
x_min -= padding; x_max += padding
y_min -= padding; y_max += padding
z_min -= padding; z_max += padding

# ----------------------------------------------------------------
# 5) Setup figure and single 3D subplot
# ----------------------------------------------------------------
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(z_min, z_max)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Skeleton Animation")

# ----------------------------------------------------------------
# 6) Create line objects for skeleton
# ----------------------------------------------------------------
def create_lines(ax, links, color):
    lines = []
    for _ in links:
        line, = ax.plot([], [], [], color=color, lw=2)
        lines.append(line)
    return lines

lines_upper = create_lines(ax, UPPER_BODY_LINKS, 'k')  # black
lines_lower = create_lines(ax, LOWER_BODY_LINKS, 'r')  # red
scat = ax.scatter([], [], [], color='magenta', s=30)

# ----------------------------------------------------------------
# 7) Initialize function for FuncAnimation
# ----------------------------------------------------------------
def init():
    for line in lines_upper + lines_lower:
        line.set_data([], [])
        line.set_3d_properties([])
    scat._offsets3d = ([], [], [])
    return lines_upper + lines_lower + [scat]

# ----------------------------------------------------------------
# 8) Update function
# ----------------------------------------------------------------
def update(frame_idx):
    pts = all_keypoints[frame_idx]  # shape (num_joints, 3)
    xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]

    # Update lines
    for line_obj, (i, j) in zip(lines_upper, UPPER_BODY_LINKS):
        line_obj.set_data([xs[i], xs[j]], [ys[i], ys[j]])
        line_obj.set_3d_properties([zs[i], zs[j]])
    for line_obj, (i, j) in zip(lines_lower, LOWER_BODY_LINKS):
        line_obj.set_data([xs[i], xs[j]], [ys[i], ys[j]])
        line_obj.set_3d_properties([zs[i], zs[j]])

    # Update scatter
    scat._offsets3d = (xs, ys, zs)

    # Update title with frame index
    ax.set_title(f"3D Skeleton (Frame {frame_ids[frame_idx]})")

    return lines_upper + lines_lower + [scat]

# ----------------------------------------------------------------
# 9) Create animation
# ----------------------------------------------------------------
ani = FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=False, interval=150)

# ----------------------------------------------------------------
# 10) Save animation as MP4
# ----------------------------------------------------------------
output_video = "C:/Users/MYSEL/Desktop/mmpose-updated/mmpose/3d_skeleton_animation.mp4"
fps = 10
writer = FFMpegWriter(fps=fps, codec="mpeg4")

ani.save(output_video, writer=writer)
print(f"Animation saved to {output_video}")

plt.show()
