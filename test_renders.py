import os
import cv2
import pybullet as p
from src.simulation.environment_odin import NBVODINEnv

def main():
    # Mock imshow and waitKey
    cv2.imshow = lambda *args, **kwargs: None
    cv2.waitKey = lambda *args, **kwargs: 1
    env = NBVODINEnv(headless=True)
    os.makedirs('test_renders', exist_ok=True)
    for i in range(20):
        env.reset()
        obs = env._get_obs()
        env._render_dashboard(0, 0)
        cv2.imwrite(f'test_renders/scene_{i}.png', cv2.cvtColor(env.last_rgb, cv2.COLOR_RGB2BGR))
        print(f'Saved scene {i}')

if __name__ == "__main__":
    main()
