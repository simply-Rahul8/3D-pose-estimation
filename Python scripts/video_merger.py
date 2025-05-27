#!/usr/bin/env python3
"""
videomerger_hardcoded.py

Merge three MP4 videos into one 1920×1080, 25fps, silent MP4,
with the input paths defined in the script.
"""

from moviepy.editor import VideoFileClip, concatenate_videoclips

# ─────── CONFIGURE YOUR FILES HERE ───────
INPUT_VIDEOS = [
    "C:/Users/MYSEL/Desktop/Pushups/Npush/01.mp4",
    "C:/Users/MYSEL/Desktop/Pushups/Npush/02.mp4",
    "C:/Users/MYSEL/Desktop/Pushups/Npush/03.mp4",
]
OUTPUT_VIDEO = "C:/Users/MYSEL/Desktop/Pushups/Npush/Npush.mp4"
# ─────────────────────────────────────────

def merge_videos_exact(input_paths, output_path):
    TARGET_SIZE = (1920, 1080)
    TARGET_FPS  = 25

    clips = []
    for path in input_paths:
        clip = VideoFileClip(path)
        # normalize to target FPS & size, drop audio
        clip = clip.set_fps(TARGET_FPS).resize(TARGET_SIZE).without_audio()
        clips.append(clip)

    final = concatenate_videoclips(clips, method="compose")

    final.write_videofile(
        output_path,
        codec="libx264",
        fps=TARGET_FPS,
        audio=False,
        preset="medium",
        ffmpeg_params=["-crf", "23"]
    )

    # clean up
    for c in clips:
        c.close()
    final.close()


if __name__ == "__main__":
    print("Merging:", INPUT_VIDEOS)
    merge_videos_exact(INPUT_VIDEOS, OUTPUT_VIDEO)
    print(f"✔ Done → {OUTPUT_VIDEO}")
