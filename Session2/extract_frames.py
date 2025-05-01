from pathlib import Path
import cv2
import os
import numpy as np

DATASET_ROOT = "images/robot"
OUTPUT_DIR = Path(__file__).parent.resolve() / DATASET_ROOT

def extract_diverse_frames(video_path, output_dir=OUTPUT_DIR,
                           frame_name = "",
                           min_diff=0.5, min_skip=100):
    """ Extracts frames from a given video.

        Parameters
        ----------
        video_path : str
            Path to the video
        min_skip : num, optional
            Minimum number of distance between frames, to avoid consecutive sampling
        min_diff : num, optional
            Minimum mean difference ratio, an additional technique to avoid similar frames
    """
    vc = cv2.VideoCapture(video_path)
    if not vc.isOpened():
        print("Error opening video file")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # take video name as frame base name
    frame_name_base = frame_name if frame_name else (video_path.split("/")[-1]).split(".")[0]

    frame_count = 0
    saved_count = 0
    last_saved_frame = None
    
    while True:
        ret, frame = vc.read()
        if not ret:
            break
            
        if frame_count % min_skip == 0:
            frame_name = output_dir / f"{frame_name_base}_{frame_count:6d}.jpg"

            if last_saved_frame is None:
                cv2.imwrite(f"{frame_name}", frame)
                last_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                diff = cv2.absdiff(gray_current, last_gray_frame)
                diff_ratio = np.mean(diff) / 255.0
                
                if diff_ratio > min_diff:
                    cv2.imwrite(f"{frame_name}", frame)
                    last_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            saved_count += 1
        
        frame_count += 1
    
    vc.release()
    print(f"Saved {saved_count} diverse frames out of {frame_count} total frames")

if __name__=="__main__":
    # extract_diverse_frames("videos/robot_video_1.mp4", frame_name="robot")
    extract_diverse_frames("videos/robot_video_2.mp4", frame_name="robot")
