import os
import time
import json
import cv2
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from PIL import Image, ExifTags

# GroundingDINO
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict

# SAM2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# BLIP-2
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image

# MiDaS (DPT)
from transformers import DPTForDepthEstimation, DPTImageProcessor

# Gradio
import gradio as gr

# --------------------------------------------------------------
# 1. 전역 설정
# --------------------------------------------------------------
DEVICE = "cpu" if torch.cuda.is_available() else "cpu"

# 모델 가중치/설정 경로
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
MIDAS_MODEL_NAME = "Intel/dpt-large"

# Threshold
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

# MiDaS 관련 카메라 및 참조 물체 기본 파라미터 (필요 시 수정)
FOCAL_LENGTH_MM = 3.29    
SENSOR_WIDTH_MM = 5.76   
SENSOR_HEIGHT_MM = 4.29

# --------------------------------------------------------------
# 2. 모델 빌드 함수
# --------------------------------------------------------------
def build_sam2_predictor():
    sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    return sam2_predictor

def build_grounding_dino():
    grounding_model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device=DEVICE
    )
    return grounding_model

def build_midas_model():
    midas_model = DPTForDepthEstimation.from_pretrained(MIDAS_MODEL_NAME)
    midas_processor = DPTImageProcessor.from_pretrained(MIDAS_MODEL_NAME)
    midas_model.eval()
    return midas_model, midas_processor

def build_blip2_model():
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model.to(DEVICE)
    return processor, model

# --------------------------------------------------------------
# 3. 유틸리티 함수
# --------------------------------------------------------------
def generate_depth_map(image_bgr: np.ndarray, midas_model, midas_processor):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    inputs = midas_processor(images=image_rgb, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = midas_model(**inputs)
        depth_map = outputs.predicted_depth.squeeze().cpu().numpy()

    h, w, _ = image_bgr.shape
    depth_map_resized = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_NEAREST)
    return depth_map_resized

def generate_relative_depth_map(image_bgr: np.ndarray, midas_model, midas_processor):
    # MiDaS 상대 깊이맵 생성 (CPU 고정)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    inputs = midas_processor(images=image_rgb, return_tensors="pt").to("cpu")
    with torch.no_grad():
        outputs = midas_model(**inputs)
        depth_map = outputs.predicted_depth.squeeze().cpu().numpy()
    h, w, _ = image_bgr.shape
    depth_map_resized = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_NEAREST)
    return depth_map_resized

def get_mask(segmentation):
    return mask_util.decode(segmentation)

def get_average_relative_depth(relative_depth_map, mask):
    ys, xs = np.where(mask > 0)
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

# EXIF 정보 추출
def extract_exif(image_path):
    img = Image.open(image_path)
    exif_data = img._getexif()

    if not exif_data:
        raise ValueError("EXIF 데이터를 찾을 수 없습니다.")

    exif = {ExifTags.TAGS[k]: v for k, v in exif_data.items() if k in ExifTags.TAGS}

    def convert_ifd_rational(value):
        """
        IFDRational 값을 float로 변환
        """
        if isinstance(value, tuple):
            return float(value[0]) / value[1] if value[1] != 0 else 0
        elif isinstance(value, (int, float)):
            return value
        else:
            return float(value)

    focal_length = exif.get("FocalLength")
    focal_length_35mm = exif.get("FocalLengthIn35mmFilm")
    image_width = exif.get("ExifImageWidth")
    image_height = exif.get("ExifImageHeight")

    # 필요한 EXIF 데이터가 없는 경우 예외 처리
    if not (focal_length and focal_length_35mm and image_width and image_height):
        raise ValueError("필요한 EXIF 데이터가 부족합니다.")

    return {
        "focal_length_mm": convert_ifd_rational(focal_length),
        "focal_length_35mm": convert_ifd_rational(focal_length_35mm),
        "image_width": image_width,
        "image_height": image_height,
    }

# 센서 크기 계산
def calculate_sensor_size(exif_info):
    focal_length_mm = exif_info["focal_length_mm"]
    focal_length_35mm = exif_info["focal_length_35mm"]

    crop_factor = focal_length_35mm / focal_length_mm
    diagonal_35mm = (36**2 + 24**2)**0.5
    sensor_diagonal = diagonal_35mm / crop_factor

    aspect_ratio_width = 4
    aspect_ratio_height = 3

    sensor_width = sensor_diagonal * (aspect_ratio_width / (aspect_ratio_width**2 + aspect_ratio_height**2)**0.5)
    sensor_height = sensor_diagonal * (aspect_ratio_height / (aspect_ratio_width**2 + aspect_ratio_height**2)**0.5)

    return sensor_width, sensor_height

def save_image_with_exif(input_image, save_path):
    """
    업로드된 이미지의 EXIF 데이터를 포함하여 저장
    """
   
    exif_data = input_image.info.get("exif")  # EXIF 데이터 가져오기

    # EXIF 데이터를 포함하여 저장
    if exif_data:
        input_image.save(save_path, exif=exif_data)
    else:
        input_image.save(save_path)  # EXIF 데이터가 없으면 일반 저장

# --------------------------------------------------------------
# 4. 전역 모델 로드
# --------------------------------------------------------------
sam2_predictor_global = build_sam2_predictor()
grounding_model_global = build_grounding_dino()
midas_model_global, midas_processor_global = build_midas_model()
blip2_processor_global, blip2_model_global = build_blip2_model()

# --------------------------------------------------------------
# 5. Gradio 추론 함수
# --------------------------------------------------------------
def inference(
    input_image,       
    text_prompt,       
    ref_class_name,    
    ref_width_mm,      
    ref_height_mm      
):
    # 1) 업로드된 이미지를 디스크에 저장 (EXIF 데이터 포함)
    os.makedirs("./upload_image", exist_ok=True)
    timestamp = int(time.time())
    save_path = f"./upload_image/user_upload_{timestamp}.jpg"
    save_image_with_exif(input_image, save_path)

    # 2) 저장된 파일 경로에서 EXIF 데이터 추출
    exif_info = extract_exif(save_path)

    # 3) 디버깅 정보 출력
    print(f"[DEBUG] EXIF 정보: {exif_info}")

    # EXIF 데이터 활용
    exif_info = extract_exif(save_path)
    sensor_width_mm, sensor_height_mm = calculate_sensor_size(exif_info)
    focal_length_mm = exif_info["focal_length_mm"]

    # 기존 설정 업데이트
    global FOCAL_LENGTH_MM, SENSOR_WIDTH_MM, SENSOR_HEIGHT_MM
    FOCAL_LENGTH_MM = focal_length_mm
    SENSOR_WIDTH_MM = sensor_width_mm
    SENSOR_HEIGHT_MM = sensor_height_mm

    # 2) GroundingDINO 로 이미지 로드 및 3) SAM2 분할 준비
    image_source, image_rgb = load_image(save_path)
    sam2_predictor_global.set_image(image_source.copy())

    # GroundingDINO 박스 예측
    boxes, confidences, labels = predict(
        model=grounding_model_global,
        image=image_rgb,
        caption=text_prompt,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )

    # SAM2로 분할
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes_xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    masks, scores, logits = sam2_predictor_global.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes_xyxy,
        multimask_output=False,
    )
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    # SAM2 시각화
    detections = sv.Detections(
        xyxy=input_boxes_xyxy,
        mask=masks.astype(bool),
        class_id=np.arange(len(labels))
    )
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(
        scene=image_source.copy(),
        detections=detections
    )
    label_annotator = sv.LabelAnnotator()
    confidences_np = confidences.numpy() if isinstance(confidences, torch.Tensor) else confidences
    labels_text = [f"{cls_name} {conf:.2f}" for cls_name, conf in zip(labels, confidences_np)]
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels_text
    )
    mask_annotator = sv.MaskAnnotator()
    annotated_frame_with_mask = mask_annotator.annotate(
        scene=annotated_frame.copy(),
        detections=detections
    )

    masked_image_rgb = cv2.cvtColor(annotated_frame_with_mask, cv2.COLOR_BGR2RGB)
    masked_image_pil = Image.fromarray(masked_image_rgb)

    # 4) BLIP-2 색상/스타일 추론
    annotations = []
    for cls_name, box_xyxy, mask_arr, score_val in zip(labels, input_boxes_xyxy, masks, scores):
        rle = mask_util.encode(np.asfortranarray(mask_arr.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("utf-8")
        annotations.append({
            "class_name": cls_name,
            "bbox": box_xyxy.tolist(),
            "segmentation": rle,
            "score": float(score_val),
        })

    for ann in annotations:
        cls_name = ann["class_name"]
        x1, y1, x2, y2 = ann["bbox"]

        cropped_img = input_image.crop((x1, y1, x2, y2))

        prompt_color = (
            f"Describe the color of the {cls_name}. "
            "Is it white, black, gray, red, blue, green, yellow, brown, beige, pink, purple, or orange?"
        )
        input_color = blip2_processor_global(images=cropped_img, text=prompt_color, return_tensors="pt").to(DEVICE)
        output_color = blip2_model_global.generate(**input_color)
        color_description = blip2_processor_global.decode(output_color[0], skip_special_tokens=True)

        prompt_style = (
            f"Describe the style of the {cls_name}. "
            "Is it modern, minimal, natural, vintage, classic, French, Nordic, industrial, lovely, Korean, or unique?"
        )
        input_style = blip2_processor_global(images=cropped_img, text=prompt_style, return_tensors="pt").to(DEVICE)
        output_style = blip2_model_global.generate(**input_style)
        style_description = blip2_processor_global.decode(output_style[0], skip_special_tokens=True)

        ann["color"] = color_description
        ann["style"] = style_description

    # 5) MiDaS로 깊이맵 생성 및 시각화
    depth_map = generate_depth_map(image_source, midas_model_global, midas_processor_global)
    depth_min, depth_max = depth_map.min(), depth_map.max()
    depth_map_norm = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)
    depth_map_vis = (depth_map_norm * 255).astype(np.uint8)
    depth_map_pil = Image.fromarray(depth_map_vis)
    
    # ----- 새로운 MiDaS 알고리즘 적용 시작 -----
    relative_depth_map = generate_relative_depth_map(image_source, midas_model_global, midas_processor_global)
    h_img, w_img, _ = image_source.shape

    # [A] 참조 물체 처리 (동적 참조)
    ref_ann = next((ann for ann in annotations if ann["class_name"].lower() == ref_class_name.lower()), None)
    if ref_ann is not None:
        ref_mask = get_mask(ref_ann["segmentation"])
        ref_px_w, ref_px_h = measure_2d_size_from_mask(ref_mask)
        ref_avg_rel_depth = get_average_relative_depth(relative_depth_map, ref_mask)

        Z_ref_estimated = pinhole_distance(
            object_pixel_size = ref_px_w,
            object_real_size  = ref_width_mm,
            focal_length_mm   = FOCAL_LENGTH_MM,
            sensor_size_mm    = SENSOR_WIDTH_MM,
            image_size_px     = w_img
        )

        scale = Z_ref_estimated / ref_avg_rel_depth if ref_avg_rel_depth != 0 else 1.0
        real_depth_map = relative_depth_map * scale

        ref_ann["width"] = ref_width_mm
        ref_ann["height"] = ref_height_mm
    else:
        real_depth_map = relative_depth_map

    # [B] 참조 물체 이외의 모든 객체에 대해 실제 크기 계산
    for ann in annotations:
        if ann["class_name"].lower() == ref_class_name.lower():
            continue

        obj_mask = get_mask(ann["segmentation"])
        obj_px_w, obj_px_h = measure_2d_size_from_mask(obj_mask)
        obj_avg_depth = get_average_relative_depth(real_depth_map, obj_mask)

        real_width = (obj_px_w * obj_avg_depth * SENSOR_WIDTH_MM) / (FOCAL_LENGTH_MM * w_img)
        real_height = (obj_px_h * obj_avg_depth * SENSOR_HEIGHT_MM) / (FOCAL_LENGTH_MM * h_img)

        ann["width"] = round(real_width, 2)
        ann["height"] = round(real_height, 2)
    # ----- 새로운 MiDaS 알고리즘 적용 끝 -----

    # 6) 테이블 생성
    results_for_table = []
    for ann in annotations:
        width = ann.get("width", 0.0)
        height = ann.get("height", 0.0)
        results_for_table.append([
            ann["class_name"],
            ann.get("color", ""),
            ann.get("style", ""),
            width,
            height
        ])

    return masked_image_pil, depth_map_pil, results_for_table, json.dumps(exif_info, indent=2)

# --------------------------------------------------------------
# 6. Gradio 인터페이스
# --------------------------------------------------------------
def create_demo():
    with gr.Blocks() as demo:
        gr.Markdown("## Grounded SAM2 + MiDaS + BLIP-2 Demo (Table Output)")
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Upload Image", type="pil")
                text_prompt = gr.Textbox(label="Text Prompt", value="monitor. tumbler.")
                ref_class_name = gr.Textbox(label="Reference Class Name", value="monitor")
                ref_width_mm = gr.Number(label="Reference Width (mm)", value=615.0)
                ref_height_mm = gr.Number(label="Reference Height (mm)", value=365.0)
                run_button = gr.Button("Run Inference")

            with gr.Column():
                masked_image_out = gr.Image(label="Masked Image")
                depth_map_out = gr.Image(label="Depth Map")
                results_table_out = gr.Dataframe(
                    headers=["Class", "Color", "Style", "Width(mm)", "Height(mm)"],
                    datatype=["str", "str", "str", "number", "number"],
                    label="Results Table"
                )
                exif_info_text = gr.Textbox(label="EXIF Information", interactive=False, lines=10)

        run_button.click(
            fn=inference,
            inputs=[input_image, text_prompt, ref_class_name, ref_width_mm, ref_height_mm],
            outputs=[masked_image_out, depth_map_out, results_table_out, exif_info_text]
        )
    return demo


if __name__ == "__main__":
    demo_app = create_demo()
    demo_app.launch()

