import os
import imageio
from utils.logging_utils import setup_logger

logger = setup_logger("video_utils")

def save_video(frames, path, fps=30):
    """
    Saves a sequence of RGB image frames (list of numpy arrays of shape HxWxC) to an MP4 video.
    """
    if not frames:
        logger.warning(f"No frames provided to save_video. Path: {path}")
        return False
        
    try:
        dir_name = os.path.dirname(os.path.abspath(path))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        # Write frames to MP4 using standard H.264 codec
        writer = imageio.get_writer(path, fps=fps, codec='libx264', pixelformat='yuv420p')
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        logger.info(f"Successfully saved diagnostic video to {path} ({len(frames)} frames)")
        return True
    except Exception as e:
        logger.error(f"Failed to write video to {path}. Error: {e}")
        return False
