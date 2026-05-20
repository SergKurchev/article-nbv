import cv2
import numpy as np
import imageio

for ep in [0, 4, 9]:
    gif_path = f"demo_output/videos/episode_{ep}.gif"
    reader = imageio.get_reader(gif_path)
    frames = list(reader)
    print(f"Total frames in {gif_path}:", len(frames))

first_frame = frames[0]
print("Frame shape:", first_frame.shape)

# Top row is hstack(cam_rgb [300], third_person_rgb [300], text_panel [400]) = 1000 width
# Bottom row is plot_resized [1000 width, 300 height]
# Let's inspect the subareas:
cam_area = first_frame[0:300, 0:300]
third_person_area = first_frame[0:300, 300:600]
text_area = first_frame[0:300, 600:1000]
plot_area = first_frame[300:600, 0:1000]

print("Cam area mean intensity:", np.mean(cam_area))
print("Third person area mean intensity:", np.mean(third_person_area))
print("Text area mean intensity:", np.mean(text_area))
print("Plot area mean intensity:", np.mean(plot_area))

# Let's count non-black pixels in the plot area (should be white/light-gray background and lines)
print("Non-black pixels in Plot Area:", np.sum(plot_area > 50))

cv2.imwrite("demo_output/videos/episode_9_frame_last.png", cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR))
print("Saved last frame to demo_output/videos/episode_9_frame_last.png")
