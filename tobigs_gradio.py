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
    model.to(DEVICE)
    return processor, model

# --------------------------------------------------------------
# 3. 유틸리티 함수
# --------------------------------------------------------------
def generate_depth_map(image_bgr: np.ndarray, midas_model, midas_processor):
    """
    MiDaS(DPT) 모델을 이용해 전체 이미지의 깊이맵을 생성합니다.
    """
    # OpenCV BGR -> RGB 변환
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    inputs = midas_processor(images=image_rgb, return_tensors="pt").to(DEVICE)

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

def compute_scale_factor_for_reference(annotations, ref_class_name, ref_width_mm, ref_height_mm):
    """
    참조 클래스(ref_class_name)의 마스크를 찾아
    (참조물체 실제 너비 / 픽셀 너비) 스케일 팩터 계산.
    """
    ref_anns = [ann for ann in annotations if ann["class_name"].lower() == ref_class_name.lower()]
    if len(ref_anns) == 0:
        return None

    ref_seg = ref_anns[0]["segmentation"]
    ref_mask = decode_mask_rle(ref_seg)
    xs = np.where(ref_mask > 0)[1]  # x 좌표

    if len(xs) == 0:
        return None

    ref_pixel_width = np.max(xs) - np.min(xs)
    if ref_pixel_width <= 0:
        return None

    # 여기서는 '가로 길이'만으로 스케일 팩터를 구함
    scale_factor = ref_width_mm / ref_pixel_width
    return scale_factor

def estimate_object_size(mask_rle, scale_factor):
    """
    객체 마스크에서 width, height (pixels) 구한 뒤
    scale_factor를 곱해 mm로 변환.
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
    input_image,       # Gradio로 업로드된 PIL.Image
    text_prompt,       # 예: "monitor. tumbler."
    ref_class_name,    # 참조 클래스명
    ref_width_mm,      # 참조 물체 실제 가로 길이(mm)
    ref_height_mm      # 참조 물체 실제 세로 길이(mm)
):
    """
    1) 업로드된 이미지를 ./upload_image 에 저장
    2) GroundingDINO 로 파일 경로 로드 & 박스 검출
    3) SAM2로 분할
    4) BLIP-2 (color, style)
    5) MiDaS (깊이맵)
    6) 표(Dataframe)로 결과 표시
    """
    # 1) 업로드된 이미지를 디스크에 저장
    os.makedirs("./upload_image", exist_ok=True)
    timestamp = int(time.time())
    save_path = f"./upload_image/user_upload_{timestamp}.png"
    input_image.save(save_path)

    # 2) GroundingDINO 로 이미지 로드
    #    load_image()는 (OpenCV BGR, Numpy RGB) 형태 반환
    image_source, image_rgb = load_image(save_path)

    # SAM2 준비
    sam2_predictor_global.set_image(image_source.copy())

    # GroundingDINO 박스 예측
    boxes, confidences, labels = predict(
        model=grounding_model_global,
        image=image_rgb,
        caption=text_prompt,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )

    # 3) SAM2 분할
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes_xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    masks, scores, logits = sam2_predictor_global.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes_xyxy,
        multimask_output=False,
    )
    # (N,1,H,W) -> (N,H,W)
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
    mask_annotator = sv.MaskAnnotator()
    annotated_frame_with_mask = mask_annotator.annotate(
        scene=annotated_frame.copy(),
        detections=detections
    )

    # OpenCV BGR -> PIL
    masked_image_rgb = cv2.cvtColor(annotated_frame_with_mask, cv2.COLOR_BGR2RGB)
    masked_image_pil = Image.fromarray(masked_image_rgb)

    # 4) BLIP-2 색상/스타일
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

        # crop: Gradio 업로드된 PIL (input_image)
        cropped_img = input_image.crop((x1, y1, x2, y2))

        # 색상 질문
        prompt_color = (
            f"Describe the color of the {cls_name}. "
            "Is it white, black, gray, red, blue, green, yellow, brown, beige, pink, purple, or orange?"
        )
        input_color = blip2_processor_global(images=cropped_img, text=prompt_color, return_tensors="pt").to(DEVICE)
        output_color = blip2_model_global.generate(**input_color)
        color_description = blip2_processor_global.decode(output_color[0], skip_special_tokens=True)

        # 스타일 질문
        prompt_style = (
            f"Describe the style of the {cls_name}. "
            "Is it modern, minimal, natural, vintage, classic, French, Nordic, industrial, lovely, Korean, or unique?"
        )
        input_style = blip2_processor_global(images=cropped_img, text=prompt_style, return_tensors="pt").to(DEVICE)
        output_style = blip2_model_global.generate(**input_style)
        style_description = blip2_processor_global.decode(output_style[0], skip_special_tokens=True)

        ann["color"] = color_description
        ann["style"] = style_description

    # 5) MiDaS로 깊이맵 생성
    depth_map = generate_depth_map(image_source, midas_model_global, midas_processor_global)
    depth_min, depth_max = depth_map.min(), depth_map.max()
    depth_map_norm = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)
    depth_map_vis = (depth_map_norm * 255).astype(np.uint8)
    depth_map_pil = Image.fromarray(depth_map_vis)

    # 6) 참조 물체로부터 크기 추정 & 표 생성
    scale_factor = compute_scale_factor_for_reference(
        annotations,
        ref_class_name=ref_class_name,
        ref_width_mm=ref_width_mm,
        ref_height_mm=ref_height_mm
    )
    if scale_factor is None:
        scale_factor = 0.0

    results_for_table = []
    for ann in annotations:
        if scale_factor > 0.0:
            width_mm, height_mm = estimate_object_size(ann["segmentation"], scale_factor)
        else:
            width_mm, height_mm = 0.0, 0.0

        results_for_table.append([
            ann["class_name"],
            ann["color"],
            ann["style"],
            round(width_mm, 2),
            round(height_mm, 2)
        ])

    # 최종 반환 (마스킹이미지, 깊이맵, 표)
    return masked_image_pil, depth_map_pil, results_for_table

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

                # 표 형태로 결과 표시
                results_table_out = gr.Dataframe(
                    headers=["Class", "Color", "Style", "Width(mm)", "Height(mm)"],
                    datatype=["str", "str", "str", "number", "number"],
                    label="Results Table"
                )

        run_button.click(
            fn=inference,
            inputs=[input_image, text_prompt, ref_class_name, ref_width_mm, ref_height_mm],
            outputs=[masked_image_out, depth_map_out, results_table_out]
        )
    return demo

if __name__ == "__main__":
    demo_app = create_demo()
    demo_app.launch()
