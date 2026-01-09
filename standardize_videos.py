
import os
import subprocess
import glob

TARGET_FPS = 30.0
VIDEO_DIR = "assets/Videos/Animation"

def get_fps(filepath):
    """Parses fps from ffprobe output."""
    try:
        cmd = [
            "ffprobe", 
            "-v", "error", 
            "-select_streams", "v:0", 
            "-show_entries", "stream=r_frame_rate", 
            "-of", "default=noprint_wrappers=1:nokey=1", 
            filepath
        ]
        output = subprocess.check_output(cmd, text=True).strip()
        if '/' in output:
            num, den = map(int, output.split('/'))
            return num / den
        return float(output)
    except Exception as e:
        print(f"Error getting FPS for {filepath}: {e}")
        return None

def standardize_video(filepath):
    current_fps = get_fps(filepath)
    if current_fps is None:
        return

    # If already close to 30fps (e.g. 29.97 or 30), skip unless specifically asked to force
    if abs(current_fps - TARGET_FPS) < 0.1:
        print(f"[SKIP] {filepath} is already ~{current_fps:.2f} fps")
        return

    print(f"[PROCESSING] {filepath}: {current_fps:.2f} fps -> {TARGET_FPS} fps")
    
    # Calculate speed factor. 
    # To go from 10fps to 30fps, we want to play 3x faster aka duration 1/3.
    # pts_multiplier = current_fps / target_fps
    # Example: 10 / 30 = 0.333. New timestamp = 0.333 * Old timestamp.
    pts_multiplier = current_fps / TARGET_FPS
    
    temp_output = filepath.replace(".mp4", "_temp.mp4")
    
    cmd = [
        "ffmpeg", "-y",
        "-i", filepath,
        "-filter:v", f"setpts={pts_multiplier}*PTS", 
        "-r", str(TARGET_FPS),
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-an", # Remove audio if any (animations are muted)
        temp_output
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.replace(temp_output, filepath)
        print(f"[DONE] Updated {filepath}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to convert {filepath}")
        if os.path.exists(temp_output):
            os.remove(temp_output)

def main():
    root_dir = os.path.abspath(VIDEO_DIR)
    videos = glob.glob(os.path.join(root_dir, "**", "*.mp4"), recursive=True)
    
    print(f"Found {len(videos)} videos in {VIDEO_DIR}")
    for vid in videos:
        standardize_video(vid)

if __name__ == "__main__":
    main()
