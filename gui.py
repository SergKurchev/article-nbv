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
    
    vision_model = NetLoader.load(arch=config.CNN_ARCHITECTURE, num_classes=config.NUM_CLASSES, vector_dim=15)
    if config.CNN_LOAD_MODE == "best" and config.CNN_MODEL_PATH.exists():
        print(f"Loading weights from {config.CNN_MODEL_PATH}")
        vision_model.load_state_dict(torch.load(config.CNN_MODEL_PATH, map_location="cpu"))
    elif config.CNN_LOAD_MODE == "best":
        print("Warning: CNN_LOAD_MODE is 'best' but no weights found. Using random initialization.")
    vision_model.eval()
    
    env = NBVEnv(render_mode="human", headless=False, no_arm=args.no_arm, vision_model=vision_model)
    agent = NBVAgent(env)
    
    obs, _ = env.reset()
    auto_mode = False
    
    # ---------------------------------------------------------
    # ИСПРАВЛЕНИЕ 1: Функция для отрисовки UI (восстанавливает текст после reset)
    # ---------------------------------------------------------
    def update_ui(env_steps, reward, acc_diff, text_ids=None):
        pos_steps = [-0.5, 0.5, 1.1]
        pos_rew   = [-0.5, 0.5, 1.0]
        pos_acc   = [-0.5, 0.5, 0.9]
        
        # Если text_ids = None (например, после reset), создаем надписи заново
        if text_ids is None:
            id1 = p.addUserDebugText(f"Steps: {env_steps}", pos_steps, textColorRGB=[1,1,1], textSize=1.5, physicsClientId=env.client_id)
            id2 = p.addUserDebugText(f"Reward: {reward:.2f}", pos_rew, textColorRGB=[1,0.2,0.2], textSize=1.5, physicsClientId=env.client_id)
            id3 = p.addUserDebugText(f"Acc Diff: {acc_diff:.2f}", pos_acc, textColorRGB=[1,1,0], textSize=1.5, physicsClientId=env.client_id)
            return id1, id2, id3
        else:
            # Иначе обновляем существующие
            id1 = p.addUserDebugText(f"Steps: {env_steps}", pos_steps, textColorRGB=[1,1,1], textSize=1.5, replaceItemUniqueId=text_ids[0], physicsClientId=env.client_id)
            id2 = p.addUserDebugText(f"Reward: {reward:.2f}", pos_rew, textColorRGB=[1,0.2,0.2], textSize=1.5, replaceItemUniqueId=text_ids[1], physicsClientId=env.client_id)
            id3 = p.addUserDebugText(f"Acc Diff: {acc_diff:.2f}", pos_acc, textColorRGB=[1,1,0], textSize=1.5, replaceItemUniqueId=text_ids[2], physicsClientId=env.client_id)
            return id1, id2, id3

    # Инициализируем UI первый раз
    text_ids = update_ui(env.step_count, 0.0, 0.0)
    
    # Переменные для ручного управления камерой со стрелочек
    cam_dist = 1.5
    cam_yaw = 50.0
    cam_pitch = -35.0
    cam_target = [0.0, 0.0, 0.2]
    
    print("GUI mode initialized.")
    print("Controls:")
    print("  'n' (Next): Single Step (Агент делает 1 шаг)")
    print("  'm' (Mode): Toggle Auto-Mode (Авто-режим)")
    print("  'r' (Reset): Reset Episode")
    print("  Мышь (Камера): Ctrl + Левый клик (Вращение), Ctrl + Колесико (Сдвиг), Скролл (Зум)")
    print("  Стрелки: Вращение камеры вокруг цели")

    fps_eye = np.array([1.5, 0.0, 0.5]) 
    fixed_dist = 1.0

    while True:
        try:
            keys = p.getKeyboardEvents(physicsClientId=env.client_id)
        except p.error:
            print("Physics server disconnected. Exiting...")
            break
        
        # ---------------------------------------------------------
        # ИСПРАВЛЕНИЕ 2: Сменили 'a' и 's' на 'm' (Mode) и 'n' (Next step)
        # ---------------------------------------------------------
        if ord('m') in keys and keys[ord('m')] == p.KEY_WAS_TRIGGERED:
            auto_mode = not auto_mode
            print(f"Auto mode: {auto_mode}")
            
        if ord('r') in keys and keys[ord('r')] == p.KEY_WAS_TRIGGERED:
            obs, _ = env.reset()
            auto_mode = False
            # ПЕРЕСОЗДАЕМ ТЕКСТ ПОСЛЕ СБРОСА СИМУЛЯЦИИ!
            text_ids = update_ui(env.step_count, 0.0, 0.0, text_ids=None)
            print("Environment Reset.")
            
        cam_info = p.getDebugVisualizerCamera(physicsClientId=env.client_id)
        cam_yaw = cam_info[8]
        cam_pitch = cam_info[9]
        
        # PyBullet отдает готовый 3D-вектор направления взгляда (индекс 5)
        cam_forward = np.array(cam_info[5]) 
        
        # Вычисляем вектор "Вправо" для стрейфа (перпендикуляр к взгляду и оси Z)
        right = np.cross(cam_forward, np.array([0, 0, 1]))
        
        # Защита от деления на ноль (если смотрим ровно в пол/небо)
        if np.linalg.norm(right) < 1e-5:
            right = np.array([0, 1, 0]) 
        else:
            right = right / np.linalg.norm(right)

        step = 0.05
        
        # ---------------------------------------------------------
        # УПРАВЛЕНИЕ: МЕНЯЕМ КООРДИНАТЫ САМОГО НАБЛЮДАТЕЛЯ (fps_eye)
        # ---------------------------------------------------------
        # Вперед/Назад (Стрелка вверх/вниз - летим строго туда, куда смотрим)
        if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
            fps_eye += cam_forward * step
        if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
            fps_eye -= cam_forward * step
            
        # Влево/Вправо (Стрейф по бокам от взгляда)
        if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
            fps_eye += right * step
        if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
            fps_eye -= right * step

        # Вверх/Вниз (Q / E - двигаемся строго по глобальной оси Z)
        if ord('q') in keys and keys[ord('q')] & p.KEY_IS_DOWN:
            fps_eye[2] += step
        if ord('e') in keys and keys[ord('e')] & p.KEY_IS_DOWN:
            fps_eye[2] -= step

        # ---------------------------------------------------------
        # МАГИЯ FPS КАМЕРЫ (КОМПЕНСАЦИЯ ЦЕЛИ)
        # ---------------------------------------------------------
        # Чтобы глаз (fps_eye) оставался на месте при вращении мышкой, 
        # мы каждый кадр перерасчитываем невидимую цель камеры.
        new_target = fps_eye + cam_forward * fixed_dist
        
        # Принудительно перезаписываем камеру
        p.resetDebugVisualizerCamera(
            cameraDistance=fixed_dist, 
            cameraYaw=cam_yaw, 
            cameraPitch=cam_pitch, 
            cameraTargetPosition=new_target.tolist(), 
            physicsClientId=env.client_id
        )

        
        action = None
        take_step = False
        
        if auto_mode:
            action, _ = agent.predict(obs, deterministic=True)
            take_step = True
            time.sleep(0.05)
        else:
            if ord('n') in keys and keys[ord('n')] == p.KEY_WAS_TRIGGERED:
                action, _ = agent.predict(obs, deterministic=True)
                take_step = True
                
        if take_step and action is not None:
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Логика расчета вероятностей (без изменений)
            rgbd = obs["image"]
            vector = obs["vector"]
            img_t = torch.FloatTensor(rgbd).unsqueeze(0).to(next(vision_model.parameters()).device)
            vec_t = torch.FloatTensor(vector).unsqueeze(0).to(next(vision_model.parameters()).device)
            
            with torch.no_grad():
                logits, mask_pred = vision_model(img_t, vec_t)
                probs = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()
                action_probs = " | ".join([f"{p:.2f}" for p in probs])
                
            print(f"Action probs: {action_probs}")
            
            acc_diff = info.get("acc_diff", 0.0)
            
            # Обновляем текст на экране с новыми данными!
            text_ids = update_ui(env.step_count, reward, acc_diff, text_ids)
            
            if terminated or truncated:
                print("Episode Done. Resetting...")
                obs, _ = env.reset()
                # При перезапуске эпизода текст опять слетит, поэтому восстанавливаем его
                text_ids = update_ui(env.step_count, 0.0, 0.0, text_ids=None)
                
        time.sleep(0.01)

if __name__ == "__main__":
    gui()