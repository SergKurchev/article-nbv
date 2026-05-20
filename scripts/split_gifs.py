import os
from pathlib import Path
from PIL import Image

def split_all_gifs(videos_dir, output_parent_dir_name="gif_frames"):
    videos_path = Path(videos_dir)
    if not videos_path.exists():
        print(f"Directory {videos_path} does not exist!")
        return

    output_parent_path = videos_path / output_parent_dir_name
    output_parent_path.mkdir(exist_ok=True)

    gif_files = list(videos_path.glob("*.gif"))
    if not gif_files:
        print(f"No GIF files found in {videos_path}")
        return

    print(f"Found {len(gif_files)} GIF files to split.")

    for gif_file in gif_files:
        gif_name = gif_file.stem
        gif_output_dir = output_parent_path / gif_name
        gif_output_dir.mkdir(exist_ok=True)

        print(f"Splitting {gif_file.name} into {gif_output_dir}...")
        
        try:
            with Image.open(gif_file) as im:
                frame_idx = 0
                while True:
                    # Save current frame as PNG
                    frame_path = gif_output_dir / f"frame_{frame_idx:02d}.png"
                    # Convert to RGB if palette image
                    frame_rgb = im.convert("RGB")
                    frame_rgb.save(frame_path)
                    
                    frame_idx += 1
                    try:
                        im.seek(im.tell() + 1)
                    except EOFError:
                        break
                print(f"  Successfully extracted {frame_idx} frames.")
        except Exception as e:
            print(f"  Error splitting {gif_file.name}: {e}")

if __name__ == "__main__":
    # Path specified by the user
    videos_dir = r"c:\Users\NeverGonnaGiveYouUp\OneDrive\Рабочий стол\study_materials\Skoltech\Reinforcement_Learning\Project 4 TEST\NBV_with_obstacles_and_robot\kaggle_output\videos"
    split_all_gifs(videos_dir)
