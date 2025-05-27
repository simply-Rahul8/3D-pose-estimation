import cv2
import numpy as np
import json

# COCO skeleton pairs (17 keypoints)
coco_skeleton = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

def draw_2d_skeleton(frame, keypoints_2d):
    """
    Draw 2D skeleton on the given frame.

    Args:
        frame (np.ndarray): BGR image (height x width x 3).
        keypoints_2d (np.ndarray): shape (17, 2), COCO order.

    Returns:
        np.ndarray: The same frame with skeleton drawn on it.
    """
    # Draw keypoints
    for idx, (x, y) in enumerate(keypoints_2d):
        cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)  # Red dot
        cv2.putText(frame, str(idx), (int(x)+3, int(y)+3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Draw skeleton lines
    for (i, j) in coco_skeleton:
        x1, y1 = keypoints_2d[i]
        x2, y2 = keypoints_2d[j]
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # Blue line

    return frame


def main():
    # Path to your JSON file with 2D keypoints in COCO order
    input_json_path = r"C:/Users/MYSEL/Desktop/mmpose-updated/mmpose/vis_results/results_yoga_video.json"

    # Load JSON
    with open(input_json_path, 'r') as f:
        data = json.load(f)

    # Extract frames info
    frames_info = data.get("instance_info", [])
    if not frames_info:
        print("No 'instance_info' found in the input JSON file.")
        return

    # Loop through each frame
    for frame_data in frames_info:
        frame_id = frame_data.get("frame_id", -1)
        instances = frame_data.get("instances", [])
        if not instances:
            continue

        # Take the first instance in this example
        kp_list = instances[0].get("keypoints", [])
        if len(kp_list) != 17:
            print(f"Warning: expected 17 keypoints, got {len(kp_list)} in frame {frame_id}.")
            continue

        # Convert to NumPy array (shape: [17, 2])
        keypoints_2d = np.array(kp_list, dtype=np.float32)

        # Create a blank image (white background); adjust size if needed
        img_height, img_width = 720, 1280
        blank_image = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

        # Draw skeleton
        output_frame = draw_2d_skeleton(blank_image, keypoints_2d)

        # Show the frame
        cv2.imshow(f"2D Skeleton (Frame {frame_id})", output_frame)
        print(f"Showing frame {frame_id}... press any key to continue, 'q' to quit.")
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
