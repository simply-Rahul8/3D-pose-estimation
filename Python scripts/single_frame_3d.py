import json
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

# Set an interactive backend for Matplotlib (so we can update frames live)
matplotlib.use('TkAgg')  # Or 'Qt5Agg', etc., if 'TkAgg' is incompatible

# ---------------------
# 1) COCO Joint -> Body Part
# ---------------------
joint_to_body_part = {
    0:  'nose',
    1:  'eyes',    # left eye
    2:  'eyes',    # right eye
    3:  'ears',    # left ear
    4:  'ears',    # right ear
    5:  'shoulders',  # left shoulder
    6:  'shoulders',  # right shoulder
    7:  'elbows',     # left elbow
    8:  'elbows',     # right elbow
    9:  'wrists',     # left wrist
    10: 'wrists',     # right wrist
    11: 'hips',       # left hip
    12: 'hips',       # right hip
    13: 'knees',      # left knee
    14: 'knees',      # right knee
    15: 'ankles',     # left ankle
    16: 'ankles',     # right ankle
}

# ---------------------
# 2) Body-Part Colors (for Joints)
# ---------------------
body_part_colors = {
    'nose':      'orange',
    'eyes':      'lime',
    'ears':      'cyan',
    'shoulders': 'red',
    'elbows':    'purple',
    'wrists':    'brown',
    'hips':      'green',
    'knees':     'blue',
    'ankles':    'navy',
    'default':   'gray'
}

# ---------------------
# 3) COCO Skeleton Connections (17 Joints, 14 Lines)
#    Each pair (i, j) indicates a connection from joint i to j.
# ---------------------
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2),      # nose -> eyes
    (1, 3), (2, 4),      # eyes -> ears
    (5, 6),              # shoulders
    (5, 7), (7, 9),      # left shoulder -> elbow -> wrist
    (6, 8), (8, 10),     # right shoulder -> elbow -> wrist
    (5, 11), (6, 12),    # shoulders -> hips
    (11, 13), (13, 15),  # left hip -> knee -> ankle
    (12, 14), (14, 16)   # right hip -> knee -> ankle
]

# ---------------------
# 4) One Distinct Color per Connection (one color per "bone")
#    Make sure you have at least as many colors as there are connections.
# ---------------------
CONNECTION_COLORS = [
    'red', 'blue', 'green', 'cyan',
    'magenta', 'yellow', 'orange', 'purple',
    'brown', 'pink', 'lime', 'navy',
    'gray', 'black'
]

def draw_3d_skeleton(ax, keypoints_3d):
    """
    Draw joints as colored scatter points (by body part),
    and connect them with individually-colored lines.
    Also expand axis limits to ensure the 3D skeleton
    isn't clipped at the edges.
    """
    ax.clear()

    if keypoints_3d is None or len(keypoints_3d) == 0:
        ax.set_title("3D Skeleton\n(No Data)")
        return

    # 1) Plot each joint as a scatter point
    for idx, (x, y, z) in enumerate(keypoints_3d):
        body_part = joint_to_body_part.get(idx, 'default')
        color     = body_part_colors.get(body_part, 'gray')
        ax.scatter(x, y, z, c=color, s=40)

    # 2) Draw skeleton lines (one color per connection)
    for conn_idx, (i, j) in enumerate(SKELETON_CONNECTIONS):
        if i < 0 or j < 0:
            continue
        if i >= len(keypoints_3d) or j >= len(keypoints_3d):
            continue

        bone_color = CONNECTION_COLORS[conn_idx % len(CONNECTION_COLORS)]

        xs = [keypoints_3d[i, 0], keypoints_3d[j, 0]]
        ys = [keypoints_3d[i, 1], keypoints_3d[j, 1]]
        zs = [keypoints_3d[i, 2], keypoints_3d[j, 2]]
        ax.plot(xs, ys, zs, color=bone_color, linewidth=2)

    # 3) Set up axis limits so there's extra space around the skeleton
    x_vals = keypoints_3d[:, 0]
    y_vals = keypoints_3d[:, 1]
    z_vals = keypoints_3d[:, 2]

    margin = 5000  # Adjust as needed for your data scale
    x_min, x_max = np.min(x_vals) - margin, np.max(x_vals) + margin
    y_min, y_max = np.min(y_vals) - margin, np.max(y_vals) + margin
    z_min, z_max = np.min(z_vals) - margin, np.max(z_vals) + margin

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])

    # Keep axes on the same scale if Matplotlib version >= 3.3
    ax.set_box_aspect((1, 1, 1))

    # 4) Axis labels, title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Skeleton (COCO)")


def main():
    """
    - Reads a video (2D) frame by frame.
    - Loads a JSON time-series of 3D keypoints, one entry per frame.
    - In a loop:
      1) Read next video frame
      2) Get that frame's 3D keypoints from JSON
      3) Display 2D frame in left subplot
      4) Display 3D skeleton in right subplot
    """

    # 1) Paths to video & JSON
    video_path = "C:/Users/MYSEL/Desktop/mmpose-updated/mmpose/vis_results/walking.mp4"
    json_path  = "C:/Users/MYSEL/Desktop/mmpose-updated/mmpose/vis_results/walking_result_time_series_3d.json"

    # 2) Load 3D Keypoints from JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    all_3d_data = data.get('all_pose_3d', [])
    if not all_3d_data:
        print("No keypoints found in JSON.")
        return

    # Convert list -> dictionary {frame_idx: keypoints_3d}
    frame_dict = {}
    for entry in all_3d_data:
        frame_idx = entry.get('frame', -1)
        kp_list   = entry.get('keypoints', [])
        if kp_list:
            kp_array = np.array(kp_list, dtype=np.float32)
            frame_dict[frame_idx] = kp_array

    # 3) Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    # 4) Set up Matplotlib figure with subplots: left for video, right for 3D
    plt.ion()
    fig = plt.figure(figsize=(10, 5))
    gs  = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1.2, 1])  # left bigger, right smaller if you like

    ax_video = fig.add_subplot(gs[0, 0])
    ax_video.set_title("2D Video")

    ax_3d = fig.add_subplot(gs[0, 1], projection='3d')
    ax_3d.view_init(elev=15, azim=45)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # end of video

        # Convert the BGR frame -> RGB for Matplotlib
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Clear and show the current frame on the left subplot
        ax_video.clear()
        ax_video.imshow(frame_rgb)
        ax_video.set_title(f"2D Video - Frame {frame_idx}")
        ax_video.axis('off')

        # Retrieve 3D keypoints for this frame_idx (if available)
        kp_3d = frame_dict.get(frame_idx, None)
        draw_3d_skeleton(ax_3d, kp_3d)

        # Update figure
        plt.tight_layout()
        plt.draw()
        plt.pause(0.05)  # small delay so the figure updates

        frame_idx += 1

        # Optional: break on key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
