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

# MiDaS (DPT)
from transformers import DPTForDepthEstimation, DPTImageProcessor
import matplotlib.pyplot as plt

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
REF_MONITOR_WIDTH_MM = 615.0  # 모니터의 실제 가로 길이
REF_MONITOR_HEIGHT_MM = 365.0 # 모니터의 실제 세로 길이

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
    # DPT는 기본적으로 CPU에 올라가 있으나,
    # 필요 시 midas_model.to(DEVICE) 해도 무방 (연산이 많을 경우 GPU 사용)
    return midas_model, midas_processor

def build_blip2_model():
    """BLIP-2 (FlanT5) 모델 빌드."""
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl")
    # 모델을 GPU로 이동
    model.to("cpu")
    return processor, model

# --------------------------------------------------------------
# 3. MiDaS(DPT) 관련 유틸 함수
# --------------------------------------------------------------
def generate_depth_map(image_bgr: np.ndarray, midas_model, midas_processor):
    """
    MiDaS(DPT) 모델을 이용해 전체 이미지의 깊이맵을 생성합니다.
    """
    # OpenCV BGR -> RGB 변환
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    inputs = midas_processor(images=image_rgb, return_tensors="pt").to("cpu")

    with torch.no_grad():
        outputs = midas_model(**inputs)
        depth_map = outputs.predicted_depth.squeeze().cpu().numpy()

    # 원본 사이즈(h, w)에 맞춰서 리사이즈
    h, w, _ = image_bgr.shape
    depth_map_resized = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_NEAREST)
    return depth_map_resized

def decode_mask_rle(rle_seg):
    """pycocotools RLE 형식을 복원하여 2D mask(ndarray)로 변환."""
    return mask_util.decode(rle_seg)

def compute_scale_factor_for_monitor(annotations):
    """
    모니터 클래스를 참조하여 (모니터 실제 너비 / 모니터 픽셀 너비) 스케일 팩터 계산.
    """
    monitor_anns = [ann for ann in annotations if ann["class_name"].lower() == "monitor"]
    if len(monitor_anns) == 0:
        return None

    # 모니터가 여러 개라면, 여기서는 첫 번째만 사용 (원하는 로직으로 수정 가능)
    monitor_seg = monitor_anns[0]["segmentation"]
    monitor_mask = decode_mask_rle(monitor_seg)
    xs = np.where(monitor_mask > 0)[1]  # x 좌표만 추출

    if len(xs) == 0:
        return None

    monitor_pixel_width = np.max(xs) - np.min(xs)
    if monitor_pixel_width <= 0:
        return None

    scale_factor = REF_MONITOR_WIDTH_MM / monitor_pixel_width
    return scale_factor

def estimate_object_size(mask_rle, scale_factor):
    """
    객체 마스크로부터 width, height (pixels)을 구하고,
    스케일 팩터를 적용해 mm로 변환.
    """
    obj_mask = decode_mask_rle(mask_rle)
    ys, xs = np.where(obj_mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return 0.0, 0.0

    width_pixels = float(np.max(xs) - np.min(xs))
    height_pixels = float(np.max(ys) - np.min(ys))

    width_mm = width_pixels * scale_factor
    height_mm = height_pixels * scale_factor

    return width_mm, height_mm

# --------------------------------------------------------------
# 4. 메인 파이프라인
# --------------------------------------------------------------
def main():
    """
    1) GroundingDINO + SAM2로 객체 검출 및 분할
    2) BLIP-2로 색상, 스타일 분석
    3) MiDaS(DPT)로 전체 이미지 기준 크기 추정
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
    # GroundingDINO의 유틸 함수
    image_source, image_rgb = load_image(IMG_PATH)  # image_source: OpenCV BGR(Numpy), image_rgb: PIL->np RGB

    sam2_predictor.set_image(image_source)  # (H, W, C) BGR로 전달
    original_bgr = cv2.imread(IMG_PATH)     # MiDaS용(원본 BGR)

    # --------------------------
    # 4-3) GroundingDINO로 박스 검출
    # --------------------------
    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image_rgb,        # RGB
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
    # (N, 1, H, W) -> (N, H, W) 로 차원 축소
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
    # confidences가 torch.Tensor인지 numpy인지 확인 -> GroundingDINO는 보통 Tensor 리턴
    if isinstance(confidences, torch.Tensor):
        confidences_np = confidences.numpy()
    else:
        confidences_np = confidences

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

    # scores가 이미 numpy 배열인 경우 대비
    if isinstance(scores, torch.Tensor):
        scores_np = scores.numpy()
    else:
        scores_np = scores

    # annotations 구성
    annotations = []
    for cls_name, box_xyxy, mask_arr, score_val in zip(labels, input_boxes_xyxy, masks, scores_np):
        # mask를 pycocotools RLE로 변환
        rle = mask_util.encode(np.asfortranarray(mask_arr.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("utf-8")  # JSON 저장 호환성

        annotations.append({
            "class_name": cls_name,
            "bbox": box_xyxy.tolist(),
            "segmentation": rle,
            "score": float(score_val),
        })

    # BLIP-2 색상/스타일 분석 및 저장용 dict 구성
    for annotation in annotations:
        cls_name = annotation["class_name"]
        x1, y1, x2, y2 = annotation["bbox"]

        cropped_img = pil_image.crop((x1, y1, x2, y2))

        # BLIP-2 입력 구성
        text_color = (
            f"Describe the color of the {cls_name}. "
            "Is it white, black, gray, red, blue, green, yellow, brown, beige, pink, purple, or orange?"
        )
        input_color = blip2_processor(images=cropped_img, text=text_color, return_tensors="pt")

        # 텐서 -> GPU 이동
        for k, v in input_color.items():
            if torch.is_tensor(v):
                input_color[k] = v.to("cpu")

        # BLIP-2 추론 (색상)
        output_color = blip2_model.generate(**input_color)
        color_description = blip2_processor.decode(output_color[0], skip_special_tokens=True)

        # BLIP-2 입력 구성 (스타일)
        text_style = (
            f"Describe the style of the {cls_name}. "
            "Is it modern, minimal, natural, vintage, classic, French, Nordic, industrial, lovely, Korean, or unique?"
        )
        input_style = blip2_processor(images=cropped_img, text=text_style, return_tensors="pt")
        for k, v in input_style.items():
            if torch.is_tensor(v):
                input_style[k] = v.to("cpu")

        # BLIP-2 추론 (스타일)
        output_style = blip2_model.generate(**input_style)
        style_description = blip2_processor.decode(output_style[0], skip_special_tokens=True)

        annotation["color"] = color_description
        annotation["style"] = style_description

    # --------------------------
    # 4-7) MiDaS로 크기 추정
    # --------------------------
    depth_map = generate_depth_map(original_bgr, midas_model, midas_processor)

    # 모니터로부터 scale_factor 구하기
    scale_factor = compute_scale_factor_for_monitor(annotations)
    if scale_factor is None:
        scale_factor = 0.0  # 모니터를 찾지 못하면 0 처리

    # 각 객체에 대해 width/height(mm) 추정
    results_for_json = []
    for ann in annotations:
        mask_rle = ann["segmentation"]

        if scale_factor > 0.0:
            width_mm, height_mm = estimate_object_size(mask_rle, scale_factor)
        else:
            width_mm, height_mm = 0.0, 0.0

        # 최종 JSON 저장 구조
        results_for_json.append({
            "class_name": ann["class_name"],
            "color": ann["color"],
            "style": ann["style"],
            "width": round(width_mm, 2),
            "height": round(height_mm, 2),
        })

    # --------------------------
    # 4-8) 결과 JSON 저장
    # --------------------------
    json_path = OUTPUT_DIR / RESULT_JSON_NAME
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_for_json, f, indent=4, ensure_ascii=False)

    print(f"[INFO] Results saved in {json_path}")
    print(f"[INFO] Annotated images saved in:")
    print(f"       - {OUTPUT_DIR / 'annotated_image.jpg'}")
    print(f"       - {OUTPUT_DIR / 'annotated_image_with_mask.jpg'}")


if __name__ == "__main__":
    main()
