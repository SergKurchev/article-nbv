"""
odin_adapter.py — Адаптер NBVActiveODIN для использования в NBVEnv (RL среде).

Назначение:
    Этот модуль позволяет использовать предобученную модель NBVActiveODIN
    (из репозитория my_odin) в качестве «vision model» внутри NBVEnv
    без каких-либо изменений в самой модели ODIN.

Принцип работы:
    1.  ODINAdapter принимает готовую модель NBVActiveODIN.
    2.  При вызове `infer(rgb, depth, pose, intrinsics)` адаптер:
        a.  Делает backprojection (RGB-D → 3D облако точек) — так же, как
            это делает StrawberryDatasetMapper в my_train_odin.py.
        b.  Прогоняет данные через ODIN и извлекает query_feats.
        c.  Возвращает p_hidden (Coverage Head) и nbv_pos/nbv_quat (NBV Head).
    3.  NBVEnv использует p_hidden как основной сигнал для награды RL-агента:
            reward = prev_p_hidden - cur_p_hidden   (уменьшение → награда)

Совместимость:
    - NBVActiveODIN из my_train_odin.py (feature/nbv_dataset_process)
    - Датасет NBV Stage 2/3 (24 класса, cameras.json dict-формат)
    - PyBullet NBVEnv (224×224 RGB-D, cameras в OpenGL-конвенции)

Запуск на Kaggle:
    Модуль рассчитан на работу внутри venv, созданного kaggle-ноутбуком.
    Тяжелые зависимости (detectron2, pytorch3d, pointops2) уже установлены.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers — backprojection без полного detectron2-пайплайна
# ---------------------------------------------------------------------------

def _build_intrinsics(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Строим матрицу 3×3 интринзики."""
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


def _depth_to_xyz(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Backproject depth map (H,W) → XYZ map (H,W,3) в координатах камеры.
    Использует OpenGL-конвенцию: вперёд = -Z, вверх = +Y.
    """
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    X_cam = (u - cx) * depth / fx
    Y_cam = -(v - cy) * depth / fy   # Flip Y for OpenGL
    Z_cam = -depth                    # Forward is -Z
    return np.stack([X_cam, Y_cam, Z_cam], axis=-1)  # (H, W, 3)


def _apply_pose(xyz_cam: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """
    Переводим XYZ из координат камеры в мировые координаты.
    pose: (4,4) матрица [R|t] (camera-to-world).
    """
    H, W = xyz_cam.shape[:2]
    pts = xyz_cam.reshape(-1, 3)             # (N, 3)
    pts_h = np.concatenate([pts, np.ones((len(pts), 1), dtype=pts.dtype)], axis=1)  # (N, 4)
    pts_world = (pose @ pts_h.T).T[:, :3]   # (N, 3)
    return pts_world.reshape(H, W, 3)


# ---------------------------------------------------------------------------
# Основной класс адаптера
# ---------------------------------------------------------------------------

class ODINAdapter:
    """
    Обёртка над NBVActiveODIN для работы внутри NBVEnv.

    Parameters
    ----------
    model : torch.nn.Module
        Экземпляр NBVActiveODIN (уже перенесённый на нужное устройство).
    device : torch.device
        Целевое устройство (cuda / cpu).
    image_size : int
        Размер изображения, на котором обучена ODIN (обычно 224).
    num_frames : int
        Количество кадров для мульти-ракурсного батча (5 для NBV Stage 2).
    """

    def __init__(
        self,
        model: "torch.nn.Module",
        device: torch.device,
        image_size: int = 224,
        num_frames: int = 5,
    ):
        self.model = model
        self.model.eval()
        self.device = device
        self.image_size = image_size
        self.num_frames = num_frames

        # Кэш для накопления кадров окна наблюдения
        self._frame_buffer: list[dict] = []

        # Последние предсказания (для RL reward)
        self.last_p_hidden: float = 1.0
        self.last_nbv_pos: Optional[np.ndarray] = None
        self.last_nbv_quat: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def reset(self):
        """Очистить буфер кадров при сбросе эпизода."""
        self._frame_buffer.clear()
        self.last_p_hidden = 1.0
        self.last_nbv_pos = None
        self.last_nbv_quat = None

    def add_frame(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        pose: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ):
        """
        Добавляем один кадр в буфер.

        Parameters
        ----------
        rgb   : (H, W, 3) uint8
        depth : (H, W) float32, в метрах
        pose  : (4, 4) camera-to-world матрица
        fx, fy, cx, cy : параметры интринзики
        """
        # Конвертация RGB в ODIN формат: (3, H, W) float tensor
        img_t = torch.as_tensor(
            rgb.transpose(2, 0, 1).copy(), dtype=torch.float32
        )  # (3, H, W)

        depth_t = torch.as_tensor(depth, dtype=torch.float32)  # (H, W)

        K = _build_intrinsics(fx, fy, cx, cy)
        intrinsics_t = torch.from_numpy(K)  # (3, 3)

        # Конвертация позы из OpenGL → ODIN конвенция (+Z вперёд)
        pose_odin = pose.copy().astype(np.float32)
        pose_odin[:3, 2] = -pose_odin[:3, 2]  # Flip Z axis
        pose_t = torch.from_numpy(pose_odin)  # (4, 4)

        self._frame_buffer.append(
            {
                "image": img_t,
                "depth": depth_t,
                "intrinsics": intrinsics_t,
                "pose": pose_t,
            }
        )

        # Храним только последние num_frames кадров
        if len(self._frame_buffer) > self.num_frames:
            self._frame_buffer.pop(0)

    @torch.no_grad()
    def infer(
        self,
        current_pos: np.ndarray,
        current_quat: np.ndarray,
    ) -> dict:
        """
        Запуск инференса ODIN на накопленных кадрах.

        Parameters
        ----------
        current_pos  : (3,) текущая позиция камеры в мировых координатах
        current_quat : (4,) текущая ориентация камеры [x, y, z, w]

        Returns
        -------
        dict с полями:
            p_hidden  : float — вероятность скрытых объектов (0..1)
            nbv_pos   : np.ndarray (3,) — рекомендуемая позиция камеры
            nbv_quat  : np.ndarray (4,) — рекомендуемая ориентация [xyzw]
            delta_p_hidden : float — насколько снизилась неопределённость
        """
        if len(self._frame_buffer) == 0:
            return self._empty_result()

        frames = self._frame_buffer
        n = len(frames)

        # Собираем тензоры батча
        images = torch.stack([f["image"] for f in frames]).to(self.device)          # (N, 3, H, W)
        depths = torch.stack([f["depth"] for f in frames]).to(self.device)          # (N, H, W)
        poses = torch.stack([f["pose"] for f in frames]).to(self.device)            # (N, 4, 4)
        intrinsics = torch.stack([f["intrinsics"] for f in frames]).to(self.device) # (N, 3, 3)

        # Текущая поза камеры
        pos_t = torch.tensor(current_pos, dtype=torch.float32, device=self.device)   # (3,)
        quat_t = torch.tensor(current_quat, dtype=torch.float32, device=self.device) # (4,)

        # Backprojection через ODIN утилиты
        try:
            from odin.modeling.backproject.backproject import backprojector_dataloader
            multi_scale_xyz, _, original_xyz_list = self._run_backprojection(
                images, depths, poses, intrinsics
            )
        except ImportError:
            logger.warning("odin.modeling.backproject not found — using fallback backprojection.")
            multi_scale_xyz = None
            original_xyz_list = None

        # Подготовка входного батча в формате ODIN
        batched_inputs = self._build_odin_input(
            images, depths, poses, intrinsics, multi_scale_xyz, original_xyz_list, pos_t, quat_t
        )

        # Прямой проход через модель
        try:
            outputs = self.model(batched_inputs)
        except Exception as e:
            logger.error(f"ODIN forward pass failed: {e}")
            return self._empty_result()

        # Извлечение Coverage/NBV предсказаний
        prev_p_hidden = self.last_p_hidden

        if "p_hidden" in outputs:
            p_hidden_logit = outputs["p_hidden"]  # Tensor (B, 1) — сырые логиты
            p_hidden = torch.sigmoid(p_hidden_logit).item()
        elif "coverage" in outputs:
            p_hidden = float(outputs["coverage"])
        else:
            p_hidden = prev_p_hidden

        nbv_pos = None
        nbv_quat = None
        if "nbv_pos" in outputs and "nbv_quat" in outputs:
            nbv_pos = outputs["nbv_pos"].cpu().numpy().squeeze()    # (3,)
            nbv_quat = outputs["nbv_quat"].cpu().numpy().squeeze()  # (4,)

        self.last_p_hidden = p_hidden
        self.last_nbv_pos = nbv_pos
        self.last_nbv_quat = nbv_quat

        return {
            "p_hidden": p_hidden,
            "nbv_pos": nbv_pos,
            "nbv_quat": nbv_quat,
            "delta_p_hidden": prev_p_hidden - p_hidden,  # положительное = хорошо
        }

    def infer_for_rl(
        self,
        current_pos: np.ndarray,
        current_quat: np.ndarray,
    ) -> dict:
        """
        Инференс ODIN **с сохранением градиентного графа** для REINFORCE.

        В отличие от `infer()`:
          - НЕ использует `@torch.no_grad()`.
          - Возвращает `nbv_pos` и `nbv_quat` как **torch.Tensor** (с grad),
            чтобы можно было вычислить `log_prob` и сделать `.backward()`.
          - `p_hidden` возвращается как float (Coverage Head замораживается
            отдельно и не нуждается в градиентах — он задаёт сигнал награды).

        Returns
        -------
        dict:
            nbv_pos_t    : Tensor (3,)  — предсказанная позиция (с grad)
            nbv_quat_t   : Tensor (4,)  — предсказанный кватернион (с grad)
            p_hidden     : float
            delta_p_hidden : float
        """
        if len(self._frame_buffer) == 0:
            return self._empty_result_rl()

        frames = self._frame_buffer
        n = len(frames)

        images = torch.stack([f["image"] for f in frames]).to(self.device)
        depths = torch.stack([f["depth"] for f in frames]).to(self.device)
        poses = torch.stack([f["pose"] for f in frames]).to(self.device)
        intrinsics = torch.stack([f["intrinsics"] for f in frames]).to(self.device)

        pos_t = torch.tensor(current_pos, dtype=torch.float32, device=self.device)
        quat_t = torch.tensor(current_quat, dtype=torch.float32, device=self.device)

        # Backprojection (без градиентов — это геометрическая операция)
        try:
            from odin.modeling.backproject.backproject import backprojector_dataloader
            with torch.no_grad():
                multi_scale_xyz, _, original_xyz_list = self._run_backprojection(
                    images, depths, poses, intrinsics
                )
        except ImportError:
            multi_scale_xyz = None
            original_xyz_list = None

        batched_inputs = self._build_odin_input(
            images, depths, poses, intrinsics, multi_scale_xyz, original_xyz_list, pos_t, quat_t
        )

        # Прямой проход **с градиентами** через модель
        # Градиенты потекут только через те параметры, у которых requires_grad=True
        try:
            outputs = self.model(batched_inputs)
        except Exception as e:
            logger.error(f"ODIN RL forward pass failed: {e}")
            return self._empty_result_rl()

        prev_p_hidden = self.last_p_hidden

        # Coverage Head — берём значение без градиента (это сигнал награды)
        if "p_hidden" in outputs:
            p_hidden = torch.sigmoid(outputs["p_hidden"]).detach().item()
        elif "coverage" in outputs:
            p_hidden = float(outputs["coverage"])
        else:
            p_hidden = prev_p_hidden

        # NBV Head — сохраняем тензоры С градиентами
        nbv_pos_t = None
        nbv_quat_t = None
        if "nbv_pos" in outputs and "nbv_quat" in outputs:
            nbv_pos_t = outputs["nbv_pos"].squeeze()     # (3,) tensor с grad
            nbv_quat_t = outputs["nbv_quat"].squeeze()   # (4,) tensor с grad

        self.last_p_hidden = p_hidden
        if nbv_pos_t is not None:
            self.last_nbv_pos = nbv_pos_t.detach().cpu().numpy()
        if nbv_quat_t is not None:
            self.last_nbv_quat = nbv_quat_t.detach().cpu().numpy()

        return {
            "nbv_pos_t": nbv_pos_t,
            "nbv_quat_t": nbv_quat_t,
            "p_hidden": p_hidden,
            "delta_p_hidden": prev_p_hidden - p_hidden,
        }

    def _empty_result_rl(self) -> dict:
        """Пустой результат для RL (тензоры-заглушки с grad)."""
        return {
            "nbv_pos_t": torch.zeros(3, device=self.device, requires_grad=True),
            "nbv_quat_t": torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device, requires_grad=True),
            "p_hidden": self.last_p_hidden,
            "delta_p_hidden": 0.0,
        }

    # ------------------------------------------------------------------
    # Внутренние вспомогательные методы
    # ------------------------------------------------------------------

    def _run_backprojection(self, images, depths, poses, intrinsics):
        """Запускаем ODIN backprojector для построения облаков точек."""
        from odin.modeling.backproject.backproject import backprojector_dataloader

        H, W = images.shape[-2], images.shape[-1]
        # Заглушки для фичей (нас интересует только XYZ)
        scales = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
        n = images.shape[0]
        # Дополнение кратно 32
        pad_h = (32 - H % 32) % 32
        pad_w = (32 - W % 32) % 32
        H_p, W_p = H + pad_h, W + pad_w

        feats = {k: torch.zeros(n, 1, H_p // s, W_p // s, device=self.device)
                 for k, s in scales.items()}

        multi_scale_xyz, _, original_xyz_list = backprojector_dataloader(
            list(feats.values()),
            depths,
            poses,
            intrinsics,
            augment=False,
            method="bilinear",
            scannet_pc=None,
            padding=(pad_h, pad_w),
            do_rot_scale=False,
        )
        return multi_scale_xyz, None, original_xyz_list

    def _build_odin_input(
        self, images, depths, poses, intrinsics, multi_scale_xyz, original_xyz_list, pos_t, quat_t
    ) -> list[dict]:
        """Формируем список батч-дикшенариев в формате detectron2."""
        from detectron2.structures import Instances

        n, _, H, W = images.shape
        image_shape = (H, W)

        instances_list = []
        for _ in range(n):
            inst = Instances(image_shape)
            inst.gt_classes = torch.zeros(0, dtype=torch.int64)
            inst.instance_ids = torch.zeros(0, dtype=torch.int64)
            inst.gt_masks = torch.zeros((0, H, W), dtype=torch.bool)
            instances_list.append(inst)

        batch_dict = {
            "image": images[0],
            "images": [img for img in images],
            "depths": [d for d in depths],
            "poses": [p for p in poses],
            "intrinsics": [k for k in intrinsics],
            "instances_all": instances_list,
            "instances": instances_list[0],
            "height": H,
            "width": W,
            "new_image_shape": (H, W),
            "padding_masks": torch.zeros((H, W), dtype=torch.bool, device=self.device),
            "decoder_3d": True,
            "do_camera_drop": False,
            "max_frames": n,
            "use_ghost": False,
            "multiplier": 1.0,
            "valids": [d > 0.001 for d in depths],
            "length": n,
            "file_name": "synthetic/scene_0001/color/00000.jpg",
            "file_names": ["synthetic/scene_0001/color/00000.jpg"] * n,
            "depth_file_names": ["synthetic/scene_0001/depth/00000.png"] * n,
            "masks_file_names": ["synthetic/scene_0001/mask/00000.png"] * n,
            # Coverage/NBV поля
            "coverage_gt": torch.tensor([0.0], dtype=torch.float32, device=self.device),
            "current_camera_position": pos_t,
            "current_camera_quaternion": quat_t,
        }

        if multi_scale_xyz is not None:
            batch_dict["multi_scale_xyz"] = multi_scale_xyz[::-1]
            batch_dict["multi_scale_p2v"] = [None] * len(multi_scale_xyz)
        if original_xyz_list is not None:
            batch_dict["original_xyz"] = original_xyz_list[0]

        return [batch_dict]

    def _empty_result(self) -> dict:
        return {
            "p_hidden": self.last_p_hidden,
            "nbv_pos": self.last_nbv_pos,
            "nbv_quat": self.last_nbv_quat,
            "delta_p_hidden": 0.0,
        }


# ---------------------------------------------------------------------------
# Фабрика — загрузка предобученной NBVActiveODIN модели
# ---------------------------------------------------------------------------

def load_nbv_active_odin(
    weights_path: str,
    cfg_path: str,
    num_classes: int = 24,
    device: Optional[torch.device] = None,
) -> Tuple["torch.nn.Module", "ODINAdapter"]:
    """
    Загружает предобученную модель NBVActiveODIN и возвращает адаптер.

    Parameters
    ----------
    weights_path : str
        Путь к .pth файлу с весами (model_final.pth или checkpoint).
    cfg_path : str
        Путь к YAML конфигу (configs/scannet_context/3d.yaml).
    num_classes : int
        Количество классов (24 для NBV Stage 2).
    device : torch.device, optional
        Целевое устройство. По умолчанию CUDA если доступен.

    Returns
    -------
    model : NBVActiveODIN — обёрнутая модель (не обучается в этот момент)
    adapter : ODINAdapter — интерфейс для RL среды
    """
    import sys, os

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загружаем конфиг через detectron2
    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    from odin import add_maskformer2_config, add_maskformer2_video_config

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_video_config(cfg)
    cfg.merge_from_file(cfg_path)

    # Применяем настройки для NBV
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes
    cfg.MODEL.NBV_ACTIVE = True
    cfg.MODEL.DECODER_3D = True
    cfg.MODEL.DEVICE = str(device)
    cfg.freeze()

    # Оборачиваем в NBVActiveODIN (из my_train_odin.py)
    # Важно: импортируем из репозитория my_odin, добавленного в sys.path
    # Делаем это ДО build_model, чтобы зарегистрировать датасеты в MetadataCatalog
    try:
        from my_train_odin import (
            NBVActiveODIN, CoverageHead, NBVHead
        )
    except ImportError:
        # Если запускаем из корня my_odin
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from my_train_odin import (
            NBVActiveODIN, CoverageHead, NBVHead
        )

    # Строим базовую ODIN модель
    from detectron2.modeling import build_model
    from detectron2.data import MetadataCatalog
    if len(cfg.DATASETS.TRAIN) > 0:
        dataset_name = cfg.DATASETS.TRAIN[0]
        meta = MetadataCatalog.get(dataset_name)
        if not hasattr(meta, "thing_classes") or not meta.thing_classes:
            meta.set(thing_classes=[f"class_{i}" for i in range(num_classes)])
    base_model = build_model(cfg)

    model = NBVActiveODIN(
        base_model=base_model,
        cfg=cfg,
    )

    # Загружаем веса
    state = torch.load(weights_path, map_location="cpu")
    if "model" in state:
        state = state["model"]

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning(f"Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    model = model.to(device)
    model.eval()

    adapter = ODINAdapter(model=model, device=device)
    logger.info(f"NBVActiveODIN loaded from {weights_path} on {device}")
    return model, adapter
