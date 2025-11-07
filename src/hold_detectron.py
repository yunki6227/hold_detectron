# -*- coding: utf-8 -*-
"""
Train/Eval/Viz for climbing hold segmentation with Detectron2 (Mask R-CNN).

Changes vs Colab:
- Removed Colab drive mount and inline pip installs
- Argparse for dataset paths & output_dir
- CPU fallback if CUDA is unavailable
- Save visualizations to disk (no cv2_imshow)
- Wrapped in if __name__ == "__main__"
"""

# ---- Colab-specific (REMOVE in repo) ----
# from google.colab import drive
# drive.mount('/content/drive', force_remount=True)
# !python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

import os
import random
import argparse
import numpy as np
import torch

from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

import cv2
from detectron2.utils.visualizer import Visualizer, ColorMode


CLASSES = ["hold", "volume", "downclimb"]  # keep order consistent with your annotations


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_remove(name):
    # (Detectron2 provides .list(); remove stale regs to avoid metadata conflicts)
    if name in DatasetCatalog.list():
        DatasetCatalog.remove(name)
    if name in MetadataCatalog.list():
        MetadataCatalog.remove(name)


def register_raw(name, json_path, img_root):
    safe_remove(name)
    register_coco_instances(name, {}, json_path, img_root)
    MetadataCatalog.get(name).thing_classes = CLASSES


def make_splits(taskA_json, taskA_root, taskB_json, taskB_root, taskC_json, taskC_root, seed=42,
                train_ratio=0.70, val_ratio=0.10):
    """Register raw sets, concatenate, shuffle deterministically, and create train/val/test splits."""
    register_raw("taskA_raw", taskA_json, taskA_root)
    register_raw("taskB_raw", taskB_json, taskB_root)
    register_raw("taskC_raw", taskC_json, taskC_root)

    taskA = DatasetCatalog.get("taskA_raw")
    taskB = DatasetCatalog.get("taskB_raw")
    taskC = DatasetCatalog.get("taskC_raw")
    all_dicts = taskA + taskB + taskC

    assert 0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1
    test_ratio = 1.0 - train_ratio - val_ratio

    rnd = random.Random(seed)
    rnd.shuffle(all_dicts)

    n = len(all_dicts)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    n_test = n - n_train - n_val

    train_dicts = all_dicts[:n_train]
    val_dicts = all_dicts[n_train:n_train + n_val]
    test_dicts = all_dicts[n_train + n_val:]

    # in-memory dataset registrations
    DatasetCatalog.register("boulder_train", lambda d=train_dicts: d)
    DatasetCatalog.register("boulder_val",   lambda d=val_dicts: d)
    DatasetCatalog.register("boulder_test",  lambda d=test_dicts: d)

    for name in ["boulder_train", "boulder_val", "boulder_test"]:
        MetadataCatalog.get(name).thing_classes = CLASSES

    print(f"[split] total={n}  train={len(train_dicts)}  val={len(val_dicts)}  test={len(test_dicts)}")
    return {"train": "boulder_train", "val": "boulder_val", "test": "boulder_test"}


def build_train_cfg(output_dir, num_classes, device="cuda"):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )

    cfg.DATASETS.TRAIN = ("boulder_train",)
    cfg.DATASETS.TEST = ("boulder_val",)

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

    # Colab-friendly / small-batch defaults
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 2.5e-4
    cfg.SOLVER.MAX_ITER = 8000
    cfg.SOLVER.STEPS = []  # no LR decay for first runs
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000

    # small-object leaning (tune if needed)
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 6000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 6000
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000

    # augment/sizes
    cfg.INPUT.MIN_SIZE_TRAIN = (512, 640, 768, 896)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 896
    cfg.INPUT.MAX_SIZE_TEST = 1333

    cfg.SOLVER.AMP.ENABLED = True

    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # device handling
    if device == "cpu" or not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"

    return cfg


def train(cfg, resume=True):
    if torch.cuda.is_available():
        try:
            print("CUDA:", torch.cuda.get_device_name(0))
        except Exception:
            print("CUDA available")
    else:
        print("CUDA not available; training on CPU.")

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=resume)
    trainer.train()
    return os.path.join(cfg.OUTPUT_DIR, "model_final.pth")


def build_eval_cfg(weights_path, num_classes, device="cuda"):
    cfg_eval = get_cfg()
    cfg_eval.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))
    cfg_eval.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg_eval.MODEL.WEIGHTS = weights_path

    # test-time knobs (match your train/test sizes)
    cfg_eval.MODEL.RPN.POST_NMS_TOPK_TEST = 3000
    cfg_eval.TEST.DETECTIONS_PER_IMAGE = 400
    cfg_eval.INPUT.MIN_SIZE_TEST = 896
    cfg_eval.INPUT.MAX_SIZE_TEST = 1333
    cfg_eval.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.30

    if device == "cpu" or not torch.cuda.is_available():
        cfg_eval.MODEL.DEVICE = "cpu"

    return cfg_eval


def eval_set(cfg_eval, dataset_name, output_dir):
    model = build_model(cfg_eval)
    DetectionCheckpointer(model).load(cfg_eval.MODEL.WEIGHTS)
    model.eval()

    out_dir = os.path.join(output_dir, f"eval_{dataset_name}")
    os.makedirs(out_dir, exist_ok=True)

    evaluator = COCOEvaluator(dataset_name, cfg_eval, True, output_dir=out_dir)
    loader = build_detection_test_loader(cfg_eval, dataset_name)
    results = inference_on_dataset(model, loader, evaluator)

    print(f"\n=== {dataset_name} ===")
    print(results)
    return results


def sample_visualizations(weights_path, output_dir, score_thresh=0.2, num_samples=3):
    # predictor on test split
    cfg_viz = get_cfg()
    cfg_viz.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))
    cfg_viz.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASSES)
    cfg_viz.MODEL.WEIGHTS = weights_path
    cfg_viz.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    if not torch.cuda.is_available():
        cfg_viz.MODEL.DEVICE = "cpu"

    predictor = DefaultPredictor(cfg_viz)
    meta = MetadataCatalog.get("boulder_test")
    dset = DatasetCatalog.get("boulder_test")

    vis_dir = os.path.join(output_dir, "viz_samples")
    os.makedirs(vis_dir, exist_ok=True)

    # try to resolve potential relative file paths against known roots
    # (not strictly necessary if your COCO json already has absolute paths)
    roots = []
    for nm in ["taskA_raw", "taskB_raw", "taskC_raw"]:
        # We can try to infer roots from Metadata (not guaranteed)
        # Safer: pass your roots explicitly if needed.
        pass

    picked = random.sample(dset, min(num_samples, len(dset)))
    for d in picked:
        img_path = d["file_name"]
        img = cv2.imread(img_path)
        if img is None:
            # if your COCO file_name is relative, adjust resolution here if needed
            print(f"[warn] could not read image: {img_path}")
            continue

        outputs = predictor(img)
        v = Visualizer(img[:, :, ::-1], metadata=meta, scale=0.8, instance_mode=ColorMode.IMAGE_BW)
        vis = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        out_path = os.path.join(vis_dir, os.path.basename(img_path))
        cv2.imwrite(out_path, vis.get_image()[:, :, ::-1])
        print("Saved:", out_path)


def parse_args():
    ap = argparse.ArgumentParser(description="Train/Eval/Viz for climbing hold segmentation (Detectron2)")
    ap.add_argument("--taskA_json", required=True, help="Path to Task A COCO json")
    ap.add_argument("--taskA_root", required=True, help="Path to Task A images directory")
    ap.add_argument("--taskB_json", required=True, help="Path to Task B COCO json")
    ap.add_argument("--taskB_root", required=True, help="Path to Task B images directory")
    ap.add_argument("--taskC_json", required=True, help="Path to Task C COCO json")
    ap.add_argument("--taskC_root", required=True, help="Path to Task C images directory")

    ap.add_argument("--output_dir", default="outputs", help="Where to store checkpoints/metrics/visualizations")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.70)
    ap.add_argument("--val_ratio", type=float, default=0.10)
    ap.add_argument("--max_iter", type=int, default=8000)

    ap.add_argument("--skip_train", action="store_true", help="Skip training (use existing model_final.pth)")
    ap.add_argument("--weights", default="", help="Optional path to existing weights (model_final.pth)")
    ap.add_argument("--viz_samples", type=int, default=3, help="How many test images to visualize")

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    # Register + split
    names = make_splits(
        args.taskA_json, args.taskA_root,
        args.taskB_json, args.taskB_root,
        args.taskC_json, args.taskC_root,
        seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio
    )

    # Train (unless skipped)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = build_train_cfg(args.output_dir, num_classes=len(CLASSES), device=device)
    cfg.SOLVER.MAX_ITER = args.max_iter

    if args.skip_train and os.path.isfile(args.weights):
        weights_path = args.weights
        print(f"[skip_train] Using existing weights: {weights_path}")
    else:
        weights_path = train(cfg, resume=True)

    # Eval (val + test)
    cfg_eval = build_eval_cfg(weights_path, num_classes=len(CLASSES), device=device)
    _ = eval_set(cfg_eval, names["val"], args.output_dir)
    _ = eval_set(cfg_eval, names["test"], args.output_dir)

    # Sample visualizations
    sample_visualizations(weights_path, args.output_dir, score_thresh=0.2, num_samples=args.viz_samples)
