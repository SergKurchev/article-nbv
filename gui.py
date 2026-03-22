import time
import torch
import pybullet as p
import numpy as np
import config

from src.simulation.environment import NBVEnv
from src.vision.models import NetLoader
from src.rl.agent import NBVAgent

def gui():
    args = config.get_args()
    
    vision_model = NetLoader.load()
    if config.CNN_LOAD_MODE == "best" and config.CNN_MODEL_PATH.exists():
        print(f"Loading weights from {config.CNN_MODEL_PATH}")
        vision_model.load_state_dict(torch.load(config.CNN_MODEL_PATH, map_location="cpu"))
    elif config.CNN_LOAD_MODE == "best":
        print("Warning: CNN_LOAD_MODE is 'best' but no weights found. Using random initialization.")
    vision_model.eval()
    
    env = NBVEnv(render_mode="human", headless=False, no_arm=args.no_arm, vision_model=vision_model)
    agent = NBVAgent(env)
    
    # load policy if requested
    
    obs, _ = env.reset()
    
    auto_mode = False
    
    debug_text_id_1 = p.addUserDebugText("Reward: 0.00", [-0.5, 0.5, 1.0], textColorRGB=[1,0,0], textSize=2, physicsClientId=env.client_id)
    debug_text_id_2 = p.addUserDebugText("Acc Diff: 0.00", [-0.5, 0.5, 0.9], textColorRGB=[1,1,0], textSize=2, physicsClientId=env.client_id)
    
    print("GUI mode initialized.")
    print("Controls:")
    print("  's': Single Step (Agent predicts action)")
    print("  'a': Toggle Auto-Mode (Continuous agent steps)")
    print("  'r': Reset Episode")
    print("  Camera: Arrows, Shift, Ctrl, Click+Drag, Scroll (Native PyBullet)")

    while True:
        try:
            keys = p.getKeyboardEvents(physicsClientId=env.client_id)
        except p.error:
            print("Physics server disconnected. Exiting...")
            break
        
        # 'a' or 'A' Toggle auto
        if (ord('a') in keys and keys[ord('a')] == p.KEY_WAS_TRIGGERED) or \
           (ord('A') in keys and keys[ord('A')] == p.KEY_WAS_TRIGGERED):
            auto_mode = not auto_mode
            print(f"Auto mode: {auto_mode}")
            
        # 'r' or 'R' Reset
        if (ord('r') in keys and keys[ord('r')] == p.KEY_WAS_TRIGGERED) or \
           (ord('R') in keys and keys[ord('R')] == p.KEY_WAS_TRIGGERED):
            obs, _ = env.reset()
            auto_mode = False
            print("Environment Reset.")
            
        action = None
        take_step = False
        
        if auto_mode:
            action, _ = agent.predict(obs, deterministic=True)
            take_step = True
            time.sleep(0.05)
        else:
            # 's' or 'S' Step single
            if (ord('s') in keys and keys[ord('s')] == p.KEY_WAS_TRIGGERED) or \
               (ord('S') in keys and keys[ord('S')] == p.KEY_WAS_TRIGGERED):
                action, _ = agent.predict(obs, deterministic=True)
                take_step = True
                
        if take_step and action is not None:
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Read probabilities manually
            # env.camera.get_image() already called in env.step() and returned in obs
            rgbd = obs["image"] # [4, 224, 224]
            vector = obs["vector"] # [15]
            
            img_t = torch.FloatTensor(rgbd).unsqueeze(0).to(next(vision_model.parameters()).device)
            vec_t = torch.FloatTensor(vector).unsqueeze(0).to(next(vision_model.parameters()).device)
            
            with torch.no_grad():
                logits, mask_pred = vision_model(img_t, vec_t)
                probs = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()
                action_probs = " | ".join([f"{p:.2f}" for p in probs])
                
            print(f"Action probs: {action_probs}")
            
            acc_diff = info.get("acc_diff", 0.0)
            
            # Update text
            p.addUserDebugText(f"Reward: {reward:.2f}", [-0.5, 0.5, 1.0], textColorRGB=[1,0,0], textSize=2, replaceItemUniqueId=debug_text_id_1, physicsClientId=env.client_id)
            p.addUserDebugText(f"Acc Diff: {acc_diff:.2f}", [-0.5, 0.5, 0.9], textColorRGB=[1,1,0], textSize=2, replaceItemUniqueId=debug_text_id_2, physicsClientId=env.client_id)
            
            if terminated or truncated:
                print("Episode Done. Resetting...")
                obs, _ = env.reset()
                
        # Small sleep
        time.sleep(0.01)

if __name__ == "__main__":
    gui()
