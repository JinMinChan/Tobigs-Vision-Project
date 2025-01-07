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


"""
--------------------------------------------------------------------------
1. 설정값(하이퍼파라미터)
--------------------------------------------------------------------------
"""
TEXT_PROMPT = "monitor. tumbler."
IMG_PATH = "test_image/size.jpg"

# SAM2 경로
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Grounding DINO 경로
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"

# 기타 파라미터
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 출력 폴더
OUTPUT_DIR = Path("result")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 결과 JSON 파일 이름
RESULT_JSON_NAME = "results.json"

# MiDaS (DPT) 모델 설정
MIDAS_MODEL_NAME = "Intel/dpt-large"  # 필요 시 다른 모델로 변경 가능
# 모니터(참조 물체)의 실제 크기(mm)
REF_MONITOR_WIDTH_MM = 615.0  # 예시: Q27G2S 모니터 가로 (mm)
REF_MONITOR_HEIGHT_MM = 365.0 # 예시: Q27G2S 모니터 세로 (mm)

"""
--------------------------------------------------------------------------
2. 모델 빌드 및 유틸 함수
--------------------------------------------------------------------------
"""
# SAM2 로드
def build_sam2_predictor():
    sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
    return SAM2ImagePredictor(sam2_model)

# GroundingDINO 로드
def build_grounding_dino():
    grounding_model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device=DEVICE
    )
    return grounding_model

# MiDaS(DPT) 로드
def build_midas_model():
    midas_model = DPTForDepthEstimation.from_pretrained(MIDAS_MODEL_NAME)
    midas_processor = DPTImageProcessor.from_pretrained(MIDAS_MODEL_NAME)
    midas_model.eval()
    return midas_model, midas_processor

# BLIP-2 로드
def build_blip2_model():
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl")
    return processor, model


"""
--------------------------------------------------------------------------
3. MiDaS 기반 크기 추정
--------------------------------------------------------------------------
"""
def generate_depth_map(image: np.ndarray, midas_model, midas_processor):
    """
    MiDaS(DPT) 모델을 이용해 전체 이미지의 깊이 맵을 생성합니다.
    """
    # BGR -> RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inputs = midas_processor(images=image_rgb, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = midas_model(**inputs)
        depth_map = outputs.predicted_depth.squeeze().cpu().numpy()

    # 원본 사이즈에 맞춰 리사이즈 (interpolation=NEAREST)
    h, w, _ = image.shape
    depth_map_resized = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_NEAREST)
    return depth_map_resized

def decode_mask_rle(rle_seg):
    """
    pycocotools 형식의 RLE( {'counts':..., 'size':...} ) 마스크를 복원합니다.
    """
    return mask_util.decode(rle_seg)

def compute_scale_factor_for_monitor(annotations, depth_map):
    """
    모니터 클래스를 참조물로 하여 스케일 팩터를 구합니다.
    - 모니터의 실제 너비(mm) / 모니터 마스크의 픽셀 너비
    """
    # 모니터 클래스만 필터
    monitor_anns = [ann for ann in annotations if ann["class_name"].lower() == "monitor"]
    if len(monitor_anns) == 0:
        # 모니터가 없으면 None 반환
        return None

    # 첫 번째 모니터만 사용 (복수개라면 추가 로직 필요)
    monitor_seg = monitor_anns[0]["segmentation"]
    monitor_mask = decode_mask_rle(monitor_seg)

    # 모니터의 x좌표 최소/최대
    xs = np.where(monitor_mask > 0)[1]
    if len(xs) == 0:
        return None

    monitor_pixel_width = np.max(xs) - np.min(xs)
    if monitor_pixel_width <= 0:
        return None

    # 모니터 실제 너비 / 모니터 픽셀 너비
    scale_factor = REF_MONITOR_WIDTH_MM / monitor_pixel_width
    return scale_factor

def estimate_object_size(mask_rle, scale_factor):
    """
    객체의 마스크로부터 width, height(pixel)를 구하고, scale_factor를 적용해 mm로 변환합니다.
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


"""
--------------------------------------------------------------------------
4. 메인 파이프라인
--------------------------------------------------------------------------
"""
def main():
    """
    1) GroundingDINO + SAM2로 객체 검출, 분할
    2) BLIP-2로 색상/스타일 분석
    3) MiDaS(DPT)로 크기 추정
    4) JSON 저장 (class, color, style, width, height)
    """
    # --------------------------
    # 4-1) 모델 준비
    # --------------------------
    sam2_predictor = build_sam2_predictor()
    grounding_dino_model = build_grounding_dino()
    midas_model, midas_processor = build_midas_model()
    blip2_processor, blip2_model = build_blip2_model()

    # --------------------------
    # 4-2) 이미지 로드
    # --------------------------
    image_source, image_rgb = load_image(IMG_PATH)  # GroundingDINO에서 제공하는 유틸
    sam2_predictor.set_image(image_source)  # (H, W, C), OpenCV BGR 형태

    # OpenCV에서 읽을 때 BGR이므로, MiDaS 용 depth 계산 시 이 이미지 사용
    original_bgr = cv2.imread(IMG_PATH)

    # --------------------------
    # 4-3) GroundingDINO로 박스 검출
    # --------------------------
    boxes, confidences, labels = predict(
        model=grounding_dino_model,
        image=image_rgb,       # RGB
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )

    # --------------------------
    # 4-4) SAM2로 분할
    # --------------------------
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes_xyxy = box_convert(
        boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy"
    ).numpy()

    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes_xyxy,
        multimask_output=False,
    )
    if masks.ndim == 4:  # (N, 1, H, W) -> (N, H, W)
        masks = masks.squeeze(1)

    # --------------------------
    # 4-5) 시각화 (Optional)
    # --------------------------
    # Box + Mask Annotator
    img_bgr = cv2.imread(IMG_PATH)
    detections = sv.Detections(
        xyxy=input_boxes_xyxy,
        mask=masks.astype(bool),
        class_id=np.arange(len(labels))
    )
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(
        scene=img_bgr.copy(), detections=detections
    )
    label_annotator = sv.LabelAnnotator()
    labels_text = [
        f"{cls_name} {conf:.2f}"
        for cls_name, conf in zip(labels, confidences.numpy())
    ]
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels_text
    )
    cv2.imwrite(str(OUTPUT_DIR / "annotated_image.jpg"), annotated_frame)

    mask_annotator = sv.MaskAnnotator()
    annotated_frame_with_mask = mask_annotator.annotate(
        scene=annotated_frame.copy(), detections=detections
    )
    cv2.imwrite(str(OUTPUT_DIR / "annotated_image_with_mask.jpg"), annotated_frame_with_mask)

    # --------------------------
    # 4-6) BLIP-2로 색상/스타일 분석
    # --------------------------
    pil_image = Image.open(IMG_PATH).convert("RGB")

    # 최종 저장할 결과
    results_for_json = []

    # numpy 로 변환
    confidences_np = confidences.numpy()
    boxes_xyxy = input_boxes_xyxy
    scores_np = scores.numpy()

    # annotations 정보를 모아서, 나중에 MiDaS 크기 계산시 재활용
    annotations = []
    for cls_name, box_xyxy, mask_arr, score_val in zip(labels, boxes_xyxy, masks, scores_np):
        # RLE 변환
        rle = mask_util.encode(
            np.asfortranarray(mask_arr.astype(np.uint8))
        )
        rle["counts"] = rle["counts"].decode("utf-8")  # json 저장을 위해 decode

        annotations.append({
            "class_name": cls_name,
            "bbox": box_xyxy.tolist(),
            "segmentation": rle,
            "score": float(score_val),
        })

    # BLIP-2 분석 및 JSON 항목 구성
    for annotation in annotations:
        cls_name = annotation["class_name"]
        x1, y1, x2, y2 = annotation["bbox"]
        cropped_img = pil_image.crop((x1, y1, x2, y2))

        # 1) 색상 물어보기
        text_color = (
            f"Describe the color of the {cls_name}. "
            "Is it white, black, gray, red, blue, green, yellow, brown, beige, pink, purple, or orange?"
        )
        input_color = blip2_processor(images=cropped_img, text=text_color, return_tensors="pt").to(DEVICE)
        output_color = blip2_model.generate(**input_color)
        color_description = blip2_processor.decode(output_color[0], skip_special_tokens=True)

        # 2) 스타일 물어보기
        text_style = (
            f"Describe the style of the {cls_name}. "
            "Is it modern, minimal, natural, vintage, classic, French, Nordic, industrial, lovely, Korean, or unique?"
        )
        input_style = blip2_processor(images=cropped_img, text=text_style, return_tensors="pt").to(DEVICE)
        output_style = blip2_model.generate(**input_style)
        style_description = blip2_processor.decode(output_style[0], skip_special_tokens=True)

        # 임시로 결과 누적
        annotation["color"] = color_description
        annotation["style"] = style_description

    # --------------------------
    # 4-7) MiDaS로 크기 추정
    # --------------------------
    depth_map = generate_depth_map(original_bgr, midas_model, midas_processor)

    # 모니터로부터 scale_factor를 구함
    scale_factor = compute_scale_factor_for_monitor(annotations, depth_map)

    # scale_factor가 존재하지 않으면(모니터가 없거나 픽셀 너비=0 등)
    # 크기 측정이 불가능하므로 0.0으로 처리
    if scale_factor is None:
        scale_factor = 0.0

    # 각 객체 크기 추정
    for annotation in annotations:
        mask_rle = annotation["segmentation"]
        width_mm, height_mm = 0.0, 0.0
        if scale_factor > 0:
            width_mm, height_mm = estimate_object_size(mask_rle, scale_factor)

        # 최종 JSON용
        results_for_json.append({
            "class_name": annotation["class_name"],
            "color": annotation["color"],
            "style": annotation["style"],
            "width": round(width_mm, 2),
            "height": round(height_mm, 2),
        })

    # --------------------------
    # 4-8) JSON 저장
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
