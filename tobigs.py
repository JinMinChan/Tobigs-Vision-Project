import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# BLIP-2
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image

# MiDaS (DPT) 및 관련 라이브러리
from transformers import DPTForDepthEstimation, DPTImageProcessor
import matplotlib.pyplot as plt
import cv2
from pycocotools.mask import decode as decode_mask

# --------------------------------------------------------------
# 1. 하이퍼파라미터 / 환경설정
# --------------------------------------------------------------
TEXT_PROMPT = "monitor. tumbler."
IMG_PATH = "test_image/size.jpg"

# SAM2 체크포인트/모델 설정
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Grounding DINO 경로
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"

# MiDaS(DPT) 모델 이름
MIDAS_MODEL_NAME = "Intel/dpt-large"  # 필요 시 다른 모델 지정 가능

# 박스/텍스트 임계값
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

# 디바이스 설정
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 결과 출력 경로
OUTPUT_DIR = Path("result")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_JSON_NAME = "results.json"

# (예시) 모니터 실제 크기 (mm)
MONITOR_WIDTH_MM = 615.0  # 모니터의 실제 가로 길이
MONITOR_HEIGHT_MM = 365.0 # 모니터의 실제 세로 길이

# 카메라 내부 파라미터 (예: 갤럭시 S23 추정)
FOCAL_LENGTH_MM = 3.29    
SENSOR_WIDTH_MM = 5.76   
SENSOR_HEIGHT_MM = 4.29  

# --------------------------------------------------------------
# 2. 모델 빌드 함수
# --------------------------------------------------------------
def build_sam2_predictor():
    """SAM2 모델 및 이미지 예측기 빌드."""
    sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    return sam2_predictor

def build_grounding_dino():
    """GroundingDINO 모델 빌드."""
    grounding_model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device=DEVICE
    )
    return grounding_model

def build_midas_model():
    """MiDaS(DPT) 모델 빌드."""
    midas_model = DPTForDepthEstimation.from_pretrained(MIDAS_MODEL_NAME)
    midas_processor = DPTImageProcessor.from_pretrained(MIDAS_MODEL_NAME)
    midas_model.eval()
    return midas_model, midas_processor

def build_blip2_model():
    """BLIP-2 (FlanT5) 모델 빌드."""
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model.to("cpu")
    return processor, model

# --------------------------------------------------------------
# 3. MiDaS(DPT) 관련 유틸 함수 (새로운 알고리즘용)
# --------------------------------------------------------------
def generate_relative_depth_map(image_bgr: np.ndarray, midas_model, midas_processor):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    inputs = midas_processor(images=image_rgb, return_tensors="pt").to("cpu")
    with torch.no_grad():
        outputs = midas_model(**inputs)
        depth_map = outputs.predicted_depth.squeeze().cpu().numpy()
    h, w, _ = image_bgr.shape
    depth_map_resized = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_NEAREST)
    return depth_map_resized

def get_mask(segmentation):
    return decode_mask(segmentation)

def get_average_relative_depth(relative_depth_map, mask):
    ys, xs = np.where(mask > 0)
    if len(ys) == 0 or len(xs) == 0:
        return 0.0
    values = relative_depth_map[ys, xs]
    return np.mean(values)

def measure_2d_size_from_mask(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contour found in mask")
    max_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(max_contour)
    (w_rot, h_rot) = rect[1]
    if w_rot < 1e-5 or h_rot < 1e-5:
        raise ValueError("Invalid minAreaRect dimension")
    width_px = max(w_rot, h_rot)
    height_px = min(w_rot, h_rot)
    return width_px, height_px

def pinhole_distance(object_pixel_size, object_real_size, focal_length_mm, sensor_size_mm, image_size_px):
    return (focal_length_mm * object_real_size * image_size_px) / (object_pixel_size * sensor_size_mm)

# --------------------------------------------------------------
# 4. 메인 파이프라인
# --------------------------------------------------------------
def main():
    """
    1) GroundingDINO + SAM2로 객체 검출 및 분할
    2) BLIP-2로 색상, 스타일 분석
    3) MiDaS(DPT)로 전체 이미지 기준 크기 추정 (새 알고리즘 적용)
    4) 결과를 JSON(class, color, style, width, height)로 저장
    """
    # --------------------------
    # 4-1) 모델 로드
    # --------------------------
    sam2_predictor = build_sam2_predictor()
    grounding_model = build_grounding_dino()
    midas_model, midas_processor = build_midas_model()
    blip2_processor, blip2_model = build_blip2_model()

    # --------------------------
    # 4-2) 이미지 로드
    # --------------------------
    image_source, image_rgb = load_image(IMG_PATH)
    sam2_predictor.set_image(image_source)
    original_bgr = cv2.imread(IMG_PATH)

    # --------------------------
    # 4-3) GroundingDINO로 박스 검출
    # --------------------------
    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image_rgb,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )

    # --------------------------
    # 4-4) SAM2로 분할
    # --------------------------
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes_xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes_xyxy,
        multimask_output=False,
    )
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    # --------------------------
    # 4-5) 시각화 (Optional)
    # --------------------------
    img_bgr = cv2.imread(IMG_PATH)
    detections = sv.Detections(
        xyxy=input_boxes_xyxy,
        mask=masks.astype(bool),
        class_id=np.arange(len(labels))
    )

    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(
        scene=img_bgr.copy(),
        detections=detections
    )

    label_annotator = sv.LabelAnnotator()
    confidences_np = confidences.numpy() if isinstance(confidences, torch.Tensor) else confidences
    labels_text = [
        f"{cls_name} {conf:.2f}"
        for cls_name, conf in zip(labels, confidences_np)
    ]

    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels_text
    )
    cv2.imwrite(str(OUTPUT_DIR / "annotated_image.jpg"), annotated_frame)

    mask_annotator = sv.MaskAnnotator()
    annotated_frame_with_mask = mask_annotator.annotate(
        scene=annotated_frame.copy(),
        detections=detections
    )
    cv2.imwrite(str(OUTPUT_DIR / "annotated_image_with_mask.jpg"), annotated_frame_with_mask)

    # --------------------------
    # 4-6) BLIP-2로 색상/스타일 분석
    # --------------------------
    pil_image = Image.open(IMG_PATH).convert("RGB")
    scores_np = scores.numpy() if isinstance(scores, torch.Tensor) else scores

    annotations = []
    for cls_name, box_xyxy, mask_arr, score_val in zip(labels, input_boxes_xyxy, masks, scores_np):
        rle = mask_util.encode(np.asfortranarray(mask_arr.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("utf-8")
        annotations.append({
            "class_name": cls_name,
            "bbox": box_xyxy.tolist(),
            "segmentation": rle,
            "score": float(score_val),
        })

    for annotation in annotations:
        cls_name = annotation["class_name"]
        x1, y1, x2, y2 = annotation["bbox"]
        cropped_img = pil_image.crop((x1, y1, x2, y2))

        text_color = (
            f"Describe the color of the {cls_name}. "
            "Is it white, black, gray, red, blue, green, yellow, brown, beige, pink, purple, or orange?"
        )
        input_color = blip2_processor(images=cropped_img, text=text_color, return_tensors="pt")
        for k, v in input_color.items():
            if torch.is_tensor(v):
                input_color[k] = v.to("cpu")
        output_color = blip2_model.generate(**input_color)
        color_description = blip2_processor.decode(output_color[0], skip_special_tokens=True)

        text_style = (
            f"Describe the style of the {cls_name}. "
            "Is it modern, minimal, natural, vintage, classic, French, Nordic, industrial, lovely, Korean, or unique?"
        )
        input_style = blip2_processor(images=cropped_img, text=text_style, return_tensors="pt")
        for k, v in input_style.items():
            if torch.is_tensor(v):
                input_style[k] = v.to("cpu")
        output_style = blip2_model.generate(**input_style)
        style_description = blip2_processor.decode(output_style[0], skip_special_tokens=True)

        annotation["color"] = color_description
        annotation["style"] = style_description

    # --------------------------
    # 4-7) MiDaS로 크기 추정 (새로운 알고리즘 적용)
    # --------------------------
    h_img, w_img, _ = original_bgr.shape
    relative_depth_map = generate_relative_depth_map(original_bgr, midas_model, midas_processor)

    # [A] 참조 물체(모니터) 처리
    monitor_ann = next((ann for ann in annotations if ann["class_name"].lower() == "monitor"), None)
    if monitor_ann is not None:
        monitor_mask = get_mask(monitor_ann["segmentation"])
        monitor_px_w, monitor_px_h = measure_2d_size_from_mask(monitor_mask)
        monitor_avg_rel_depth = get_average_relative_depth(relative_depth_map, monitor_mask)
        Z_monitor_estimated = pinhole_distance(
            object_pixel_size = monitor_px_w,
            object_real_size  = MONITOR_WIDTH_MM,
            focal_length_mm   = FOCAL_LENGTH_MM,
            sensor_size_mm    = SENSOR_WIDTH_MM,
            image_size_px     = w_img
        )
        scale = Z_monitor_estimated / monitor_avg_rel_depth if monitor_avg_rel_depth != 0 else 1.0
        real_depth_map = relative_depth_map * scale

        # 모니터 실제 크기를 참조값으로 저장
        monitor_ann["width"] = MONITOR_WIDTH_MM
        monitor_ann["height"] = MONITOR_HEIGHT_MM
    else:
        real_depth_map = relative_depth_map


    # [B] 텀블러 또는 다른 객체 실제 크기 계산
    tumbler_ann = next((ann for ann in annotations if ann["class_name"].lower() == "tumbler"), None)
    if tumbler_ann is not None:
        tumbler_mask = get_mask(tumbler_ann["segmentation"])
        tumbler_px_w, tumbler_px_h = measure_2d_size_from_mask(tumbler_mask)
        tumbler_avg_depth = get_average_relative_depth(real_depth_map, tumbler_mask)

        tumbler_real_width = (
            tumbler_px_w * tumbler_avg_depth * SENSOR_WIDTH_MM
        ) / (FOCAL_LENGTH_MM * w_img)

        tumbler_real_height = (
            tumbler_px_h * tumbler_avg_depth * SENSOR_HEIGHT_MM
        ) / (FOCAL_LENGTH_MM * h_img)

        tumbler_ann["width"] = round(tumbler_real_width, 2)
        tumbler_ann["height"] = round(tumbler_real_height, 2)
    else:
        pass

    for ann in annotations:
        if "width" not in ann or "height" not in ann:
            ann["width"] = 0.0
            ann["height"] = 0.0

    # --------------------------
    # 4-8) 결과 JSON 저장
    # --------------------------
    results_for_json = []
    for ann in annotations:
        results_for_json.append({
            "class_name": ann["class_name"],
            "color": ann.get("color", ""),
            "style": ann.get("style", ""),
            "width": ann.get("width", 0.0),
            "height": ann.get("height", 0.0),
        })

    json_path = OUTPUT_DIR / RESULT_JSON_NAME
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_for_json, f, indent=4, ensure_ascii=False)

    print(f"[INFO] Results saved in {json_path}")
    print(f"[INFO] Annotated images saved in:")
    print(f"       - {OUTPUT_DIR / 'annotated_image.jpg'}")
    print(f"       - {OUTPUT_DIR / 'annotated_image_with_mask.jpg'}")

if __name__ == "__main__":
    main()

