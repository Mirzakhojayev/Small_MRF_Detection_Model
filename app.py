import os
import gradio as gr
import torch
import cv2
import numpy as np
import yaml
from PIL import Image
from model import build_model


def load_checkpoint(model, ckpt_path: str, device: torch.device) -> None:
    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "ema_state_dict" in ckpt:
        state_dict = ckpt["ema_state_dict"]
        print("[load_checkpoint] loaded EMA weights")
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        print("[load_checkpoint] loaded model_state_dict (no EMA found)")
    else:
        state_dict = ckpt
        print("[load_checkpoint] loaded checkpoint as-is")

    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            clean_state_dict[k[7:]] = v
        else:
            clean_state_dict[k] = v
    state_dict = clean_state_dict

    cur = model.state_dict()
    ok = {k: v for k, v in state_dict.items() if k in cur and v.shape ==
          cur[k].shape}
    model.load_state_dict(ok, strict=False)
    print(
        f"[load_checkpoint] Loaded {len(ok)}/{len(cur)} parameter tensors into the model.")


# Initialize model
config_path = "config.yaml"
ckpt_path = "checkpoint_best.pth" if os.path.exists("checkpoint_best.pth") else "checkpoints/checkpoint_best.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Please place checkpoint_best.pth in the project root.")

with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

try:
    ckpt_cpu = torch.load(ckpt_path, map_location="cpu")
    state_dict_cpu = ckpt_cpu.get(
        "ema_state_dict", ckpt_cpu.get("model_state_dict", ckpt_cpu))
    reg_key = "head.reg_heads.0.6.weight"
    if reg_key in state_dict_cpu:
        out_ch = state_dict_cpu[reg_key].shape[0]
        detected_reg_max = out_ch // 4
        print(
            f"Auto-detect: detected reg_max={detected_reg_max} from checkpoint.")
        cfg["model"]["reg_max"] = detected_reg_max
except Exception as e:
    print(f"Warning: Could not pre-scan checkpoint for reg_max: {e}")

# Build model
model = build_model(cfg).to(device)
if hasattr(model, "reg_max") and "reg_max" in cfg["model"]:
    model.reg_max = cfg["model"]["reg_max"]

load_checkpoint(model, ckpt_path, device)
model.eval()


def predict(input_image, conf_threshold):
    if input_image is None:
        return None, []

    img_bgr = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
    orig_h, orig_w, _ = img_bgr.shape

    img_size = cfg["model"].get("img_size", 640)
    img_resized = cv2.resize(
        img_bgr, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    img_tensor = torch.from_numpy(
        img_rgb / 255.0).permute(2, 0, 1).float().unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        preds = model(img_tensor, conf_threshold=conf_threshold)

    annotated_img = np.array(input_image).copy()
    detections = []

    if len(preds) > 0 and len(preds[0]["boxes"]) > 0:
        boxes = preds[0]["boxes"].cpu().numpy()
        scores = preds[0]["scores"].cpu().numpy()

        for box, score in zip(boxes, scores):
            x1 = int(box[0] * orig_w / float(img_size))
            y1 = int(box[1] * orig_h / float(img_size))
            x2 = int(box[2] * orig_w / float(img_size))
            y2 = int(box[3] * orig_h / float(img_size))

            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            label = f"damage {score:.2f}"
            cv2.putText(annotated_img, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            detections.append({
                "label": "damage",
                "confidence": float(score),
                "box_xyxy": [x1, y1, x2, y2]
            })

    return annotated_img, detections


# Build Gradio Web Interface
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Input Image"),
        gr.Slider(minimum=0.01, maximum=1.0, value=0.15,
                  step=0.01, label="Confidence Threshold")
    ],
    outputs=[
        gr.Image(type="numpy", label="Detections"),
        gr.JSON(label="Prediction Details (JSON)")
    ],
    title="MRFDet Defect Detection",
    description="Upload vehicle panel photos to automatically locate scratches and dents using the Multi-Receptive Field (MRFDet) model.",
    examples=[
        [os.path.join("../coco_dataset/images", f)
         if os.path.exists("../coco_dataset/images") else None, 0.15]
        for f in (os.listdir("../coco_dataset/images")[:3] if os.path.exists("../coco_dataset/images") else [])
    ]
)

if __name__ == "__main__":
    import time
    import sys
    try:
        demo.launch(server_name="127.0.0.1", server_port=7860)
        while True:
            time.sleep(1)
    except SystemExit as e:
        print(f"Intercepted SystemExit: {e}. Keeping server thread alive...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    except KeyboardInterrupt:
        print("Stopping server...")
