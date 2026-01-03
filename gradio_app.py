
import gradio as gr
import cv2
import torch
from torch.amp import autocast
import numpy as np
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
import pandas as pd
import random
import time
from datetime import datetime
from collections import deque


CKPT_PATH          = "/mnt/d/flame_split_models/Unet_r34_512_all.ckpt"
# CKPT_PATH          = /mnt/d/flame_split/models/Unet_r50_3loss_aug_640s.ckpt"
DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INFER_SIZE         = (512, 512)   
MASK_THRESH        = 0.015
AREA_RATIO_THRESH  = 0.5  
CONSEC_FRAMES      = 10 # 15
DUMMY_LAT_RANGE    = (0, 90)
DUMMY_LON_RANGE    = (0, 180)
DUMMY_H_RANGE      = (10, 200)
INTERVAL = 3
W_AREA      = 0.4
W_PROB      = 0.4
W_PERSIST   = 0.2
EVENT_ON    = 0.6
EVENT_OFF   = 0.4

class FlameModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = smp.create_model(
            "Unet", encoder_name="resnet34",
            in_channels=3, classes=1, encoder_weights="imagenet"
        )
        params = smp.encoders.get_preprocessing_params("resnet34")
        self.register_buffer("std",  torch.tensor(params["std"]).view(1,3,1,1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1,3,1,1))
    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.model(x)

model = FlameModel.load_from_checkpoint(CKPT_PATH).to(DEVICE).half()

model = torch.compile(model.to(DEVICE))

def stream_segment_video(video_file):
    cap = cv2.VideoCapture(video_file)
    
    history = deque(maxlen=CONSEC_FRAMES)
    detecting   = False
    alerts      = []
    last_alerts_time = -float('inf')
    last_frame_time = time.time()
    fps_ema         = 0.0
    alpha_fps       = 0.1

    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        
        frame = cv2.resize(frame, INFER_SIZE, interpolation=cv2.INTER_LINEAR)

        now_fps = time.time()
        inst_fps = 1.0 / (now_fps - last_frame_time + 1e-8)
        fps_ema = alpha_fps * inst_fps + (1 - alpha_fps) * fps_ema
        last_frame_time = now_fps
        # print(f"[FPS] = {fps_ema:.1f}")

        h, w = frame.shape[:2]

        small = cv2.resize(frame, INFER_SIZE)
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        tensor= torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).to(DEVICE)

        with torch.inference_mode(), autocast(device_type="cuda", enabled=True):
            tensor_half = tensor.half()
            logits = model(tensor_half)
            prob   = torch.sigmoid(logits.float())[0,0].cpu().numpy()

        mask = (prob > MASK_THRESH).astype(np.uint8)
        mask = cv2.resize(mask, (w,h), interpolation=cv2.INTER_NEAREST)
        colored_mask = np.zeros_like(frame)
        colored_mask[mask == 1] = (0, 0, 255)
        out = cv2.addWeighted(frame, 0.7, colored_mask, 0.3, 0)

        area_ratio = 100.0 * mask.sum() / (h*w)
        mean_prob = float(prob.mean())
        
        is_hot = area_ratio >= AREA_RATIO_THRESH
        history.append(1 if is_hot else 0)
        persist_factor = sum(history) / CONSEC_FRAMES

        norm_area_ratio = min(area_ratio / AREA_RATIO_THRESH, 1.0)
        norm_prob = min(mean_prob  / MASK_THRESH, 1.0)

        score = W_AREA * norm_area_ratio + W_PROB * mean_prob + W_PERSIST * persist_factor

        # print(f"raw area ratio = {area_ratio:.2f}%   normalized = {norm_area_ratio:.2f}")
        # print(f"current mean prob: {norm_prob:.2f}")
        # print(f"current persist factor: {persist_factor:.2f}")
        # print(f"current score: {score:.2f}")

        # Normalize area, model probability(certainty), and persistence (alert at 0.6, reset at 0.4)
        now = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        sys_clock = datetime.now().isoformat(timespec="seconds")
        # hysteresis on detecting
        if not detecting and score >= EVENT_ON:
            detecting = True
            last_alerts_time = now
            alerts.append({
                # "timestamp_s": round(now, 2),
                "timestamp_s": sys_clock,
                "latitude":    round(random.uniform(*DUMMY_LAT_RANGE), 4),
                "longitude":   round(random.uniform(*DUMMY_LON_RANGE), 4),
                "height_m":    round(random.uniform(*DUMMY_H_RANGE), 1),
                "area_px":     int(mask.sum()),
            })
        elif detecting and now - last_alerts_time > INTERVAL:
            last_alerts_time = now
            alerts.append({
                # "timestamp_s": round(now, 2),
                "timestamp_s": sys_clock,
                "latitude":    round(random.uniform(*DUMMY_LAT_RANGE), 4),
                "longitude":   round(random.uniform(*DUMMY_LON_RANGE), 4),
                "height_m":    round(random.uniform(*DUMMY_H_RANGE), 1),
                "area_px":     int(mask.sum()),
            })
        elif detecting and score <= EVENT_OFF:
            detecting = False
            last_alerts_time = -float('inf')

        if detecting:
            cv2.rectangle(out, (0,0), (50,50), (0,0,255), -1)

        HEADERS = ["timestamp_s","latitude","longitude","height_m","area_px","score"]
        # yield results
        out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        if not alerts:
            df = pd.DataFrame([], columns=HEADERS)
        else:
            df = (
                pd.DataFrame(alerts).drop_duplicates(subset="timestamp_s", keep="last").sort_values("timestamp_s").reset_index(drop=True)
            )
        yield out_rgb, df

    cap.release()

with gr.Blocks() as demo:
    gr.Markdown("## Flame Segmentation Demo")

    with gr.Row():
        video_in = gr.File(label="Upload Video")
        run_btn   = gr.Button("Run Demo")

    with gr.Row():
        frame_out = gr.Image(label="Inference", width="512px", height="512px")
        tbl_out   = gr.Dataframe(
            headers=["timestamp_s","latitude","longitude","height_m","area_px"],
            label="Alerts Metadata"
        )

    run_btn.click(
        fn=stream_segment_video,
        inputs=video_in,
        outputs=[frame_out, tbl_out],
    )

demo.launch()
