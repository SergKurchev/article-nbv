import sys
import os
import torch

sys.path.insert(0, r"scratch/my_odin")

# Load config
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from odin import add_maskformer2_config, add_maskformer2_video_config

cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
add_maskformer2_video_config(cfg)
cfg.merge_from_file(r"scratch/my_odin/configs/scannet_context/3d.yaml")

cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 24
cfg.MODEL.NBV_ACTIVE = True
cfg.MODEL.DECODER_3D = True
cfg.MODEL.DEVICE = "cpu"

from my_train_odin import NBVActiveODIN
from detectron2.modeling import build_model
from detectron2.data import MetadataCatalog

dataset_name = cfg.DATASETS.TRAIN[0] if len(cfg.DATASETS.TRAIN) > 0 else "scannet"
meta = MetadataCatalog.get(dataset_name)
meta.set(thing_classes=[f"class_{i}" for i in range(24)])

print("Building base model...")
try:
    base_model = build_model(cfg)
    model = NBVActiveODIN(base_model=base_model, cfg=cfg)
    print("\nModel parameters:")
    for name, param in model.named_parameters():
        if "transformer_self_attention_layers" in name or "transformer_ffn_layers" in name or "class_embed" in name:
            print(f"  {name} -> shape {list(param.shape)}")
except Exception as e:
    print(f"Error: {e}")
