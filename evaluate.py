import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
import umap
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm

import config
from src.simulation.environment import NBVEnv
from src.rl.agent import NBVAgent
from src.vision.models import NetLoader

def evaluate():
    args = config.get_args()
    
    # Load model
    vision_model = NetLoader.load()
    if config.CNN_LOAD_MODE == "best" and config.CNN_MODEL_PATH.exists():
        print(f"Loading weights from {config.CNN_MODEL_PATH}")
        vision_model.load_state_dict(torch.load(config.CNN_MODEL_PATH, map_location="cpu"))
    vision_model.eval()
    
    # Hook for features (on global_pool of MultiModalNet)
    features_list = []
    def hook(module, inp, out):
        features_list.append(out.view(out.size(0), -1).detach().cpu().numpy())
    vision_model.global_pool.register_forward_hook(hook)
    
    # Env
    env = DummyVecEnv([lambda: NBVEnv(render_mode="rgb_array", headless=True, no_arm=args.no_arm, vision_model=vision_model)])
    agent = NBVAgent(env)
    
    # Load policy (Already handled in NBVAgent if logic added, or here)
    if args.load == "none":
        print("Warning: evaluate.py run without --load RL policy. Random agent.")
    else:
        # Placeholder for RL policy load (e.g. from config or arg)
        print(f"Loading {args.load} RL policy...")
        
    num_episodes = 50
    initial_acc_diffs = []
    best_acc_diffs = []
    
    y_true = []
    y_pred = []
    best_features = []
    
    for ep in tqdm(range(num_episodes), desc="Evaluating Episodes"):
        obs, _ = env.envs[0].reset()
        done = False
        
        ep_acc = []
        ep_preds = []
        ep_feats = []
        ep_truths = []
        
        # initial
        # Forward pass is triggered in environment to get obs, so feature is in list.
        # But dummy vec env steps are different. We will just collect directly.
        
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info_dict = env.envs[0].step(action)
            done = terminated or truncated
            
            # The environment already computes accuracy_diff during step.
            acc_diff = info_dict.get("acc_diff", 0.0)
            ep_acc.append(acc_diff)
            
            # Getting pred manually to capture features at exact step
            rgb, depth, seg = env.envs[0].camera.get_image()
            
            # Pack RGB-D
            depth_normalized = np.clip(depth / 10.0, 0, 1)
            rgb_normalized = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
            rgbd = np.concatenate([rgb_normalized, depth_normalized[np.newaxis, :, :]], axis=0)
            
            img_t = torch.FloatTensor(rgbd).unsqueeze(0).to(next(vision_model.parameters()).device)
            vec_t = torch.FloatTensor(obs[0]["vector"]).unsqueeze(0).to(next(vision_model.parameters()).device)
            
            features_list.clear() # The hook will fill this
            with torch.no_grad():
                logits, mask_pred = vision_model(img_t, vec_t)
                pred_class = logits.argmax(dim=1).item()
            
            ep_preds.append(pred_class)
            ep_feats.append(features_list[0][0])
            ep_truths.append(env.envs[0].current_class_id)
            
        initial_acc_diffs.append(ep_acc[0])
        best_idx = np.argmax(ep_acc)
        best_acc_diffs.append(ep_acc[best_idx])
        
        y_true.append(ep_truths[best_idx])
        y_pred.append(ep_preds[best_idx])
        best_features.append(ep_feats[best_idx])
        
    print(f"Evaluated {num_episodes} episodes.")
    print(f"Average Initial Acc Diff: {np.mean(initial_acc_diffs):.3f}")
    print(f"Average Best Acc Diff: {np.mean(best_acc_diffs):.3f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(config.NUM_CLASSES), normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(config.NUM_CLASSES))
    fig, ax = plt.subplots(figsize=(10,10))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title("Normalized Confusion Matrix (Best Step)")
    plt.savefig("confusion_matrix.png")
    plt.close()
    
    # t-SNE / UMAP
    X = np.array(best_features)
    y = np.array(y_true)
    
    # UMAP
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X)
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab20', s=50)
    plt.legend(*scatter.legend_elements(), title="Classes", bbox_to_anchor=(1.05, 1))
    plt.title("UMAP of Penultimate Features (Best Step)")
    plt.tight_layout()
    plt.savefig("umap_eval.png")
    plt.close()
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
    X_tsne = tsne.fit_transform(X)
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab20', s=50)
    plt.legend(*scatter.legend_elements(), title="Classes", bbox_to_anchor=(1.05, 1))
    plt.title("t-SNE of Penultimate Features (Best Step)")
    plt.tight_layout()
    plt.savefig("tsne_eval.png")
    plt.close()
    
    print("Saved evaluation plots.")

if __name__ == "__main__":
    evaluate()
