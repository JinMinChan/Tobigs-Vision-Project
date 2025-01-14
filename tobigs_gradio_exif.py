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
import pandas as pd
import gradio as gr

# GroundingDINO
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict

# SAM2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# BLIP-2
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# MiDaS (DPT)
from transformers import DPTForDepthEstimation, DPTImageProcessor

# --------------------------------------------------------------
# 1. 전역 설정 및 추천 관련 초기화
# --------------------------------------------------------------
DEVICE = "cpu" if torch.cuda.is_available() else "cpu"

SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
MIDAS_MODEL_NAME = "Intel/dpt-large"

BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

FOCAL_LENGTH_MM = 3.29    
SENSOR_WIDTH_MM = 5.76   
SENSOR_HEIGHT_MM = 4.29

# CSV 및 JSON 데이터 로드
csv_file_path = './recommend/recommend.csv'
csv_data = pd.read_csv(csv_file_path, encoding='utf-8')
json_data = [
    ["table", "modern", 800, 400],
    ["bed", "modern", 2000, 1500]
]

# 전역 변수: 마지막으로 계산된 Masked Image 저장
last_masked_image = None
stored_masked_image = None

def extract_style_simple(prompt):
    styles = {
        "modern": ["modern", "모던"],
        "natural": ["natural", "내추럴"],
        "lovely": ["lovely", "러블리"]
    }
    for key, keywords in styles.items():
        for keyword in keywords:
            if keyword in prompt.lower():
                return key
    return "unknown"

def recommend_furniture(style, json_data, csv_data):
    recommendations = []
    for item in json_data:
        name, current_style, width, height = item
        if current_style != style:
            if name.lower() == "table":
                filtered_data = csv_data[
                    (csv_data['style'].str.lower() == style) & 
                    (csv_data['category'].str.lower() == "table")
                ].copy()
                if not filtered_data.empty:
                    filtered_data.loc[:, 'size_diff'] = abs(filtered_data['size'] - width)
                    recommended_item = filtered_data.sort_values(by='size_diff').iloc[0].to_dict()
                    recommendations.append({
                        "current_item": item,
                        "recommended_item": recommended_item
                    })
            elif name.lower() == "bed":
                filtered_data = csv_data[
                    (csv_data['style'].str.lower() == style) & 
                    (csv_data['category'].str.lower() == "bed")
                ].copy()
                if not filtered_data.empty:
                    filtered_data.loc[:, 'size_diff'] = abs(filtered_data['size'] - height)
                    recommended_item = filtered_data.sort_values(by='size_diff').iloc[0].to_dict()
                    recommendations.append({
                        "current_item": item,
                        "recommended_item": recommended_item
                    })
    return recommendations

def recommend_interface(user_prompt):
    extracted_style = extract_style_simple(user_prompt)
    recommendations_text = ""
    if extracted_style != "unknown":
        recommendations = recommend_furniture(extracted_style, json_data, csv_data)
        
        rec_by_cat = {}
        for rec in recommendations:
            cat = rec['current_item'][0]
            rec_by_cat.setdefault(cat, []).append(rec)
        
        for cat, recs in rec_by_cat.items():
            recommendations_text += f"## {cat.upper()}\n\n"
            for rec in recs[:3]:
                rec_item = rec['recommended_item']
                title = rec_item.get('title', 'N/A')
                url = rec_item.get('url', None)
                # CSV의 URL 사용
                link = url if url else f"https://www.google.com/search?tbm=isch&q={title}"
                
                # cost 필드 사용
                cost = rec_item.get('cost', 'N/A')
                
                # 로컬 이미지 경로 설정
                image_path = f"./google_image/{title}.jpg"
                image_md = f"![{title}]({image_path})\n" if os.path.exists(image_path) else ""
                
                recommendations_text += f"{image_md}- **Title:** [{title}]({link}) - Cost: {cost}\n"
            recommendations_text += "\n"
        
        if not recommendations:
            recommendations_text = "No recommendations found for the selected style."
    else:
        recommendations_text = "No valid style detected in the prompt."
    return recommendations_text




# --------------------------------------------------------------
# 2. 모델 빌드 함수
# --------------------------------------------------------------
def build_sam2_predictor():
    sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
    return SAM2ImagePredictor(sam2_model)

def build_grounding_dino():
    return load_model(
        model_config_path=GROUNDING_DINO_CONFIG,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device=DEVICE
    )

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
    return cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_NEAREST)

def generate_relative_depth_map(image_bgr: np.ndarray, midas_model, midas_processor):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    inputs = midas_processor(images=image_rgb, return_tensors="pt").to("cpu")
    with torch.no_grad():
        outputs = midas_model(**inputs)
        depth_map = outputs.predicted_depth.squeeze().cpu().numpy()
    h, w, _ = image_bgr.shape
    return cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_NEAREST)

def get_mask(segmentation):
    return mask_util.decode(segmentation)

def get_average_relative_depth(relative_depth_map, mask):
    ys, xs = np.where(mask > 0)
    return np.mean(relative_depth_map[ys, xs])

def measure_2d_size_from_mask(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contour found in mask")
    max_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(max_contour)
    w_rot, h_rot = rect[1]
    if w_rot < 1e-5 or h_rot < 1e-5:
        raise ValueError("Invalid minAreaRect dimension")
    return max(w_rot, h_rot), min(w_rot, h_rot)

def pinhole_distance(object_pixel_size, object_real_size, focal_length_mm, sensor_size_mm, image_size_px):
    return (focal_length_mm * object_real_size * image_size_px) / (object_pixel_size * sensor_size_mm)

def extract_exif(image_path):
    img = Image.open(image_path)
    width, height = img.size  # 이미지의 너비와 높이 가져오기
    try:
        exif_data = img._getexif()
        if not exif_data:
            raise ValueError("EXIF 데이터를 찾을 수 없습니다.")
        exif = {ExifTags.TAGS[k]: v for k, v in exif_data.items() if k in ExifTags.TAGS}
        
        def convert_ifd_rational(value):
            if isinstance(value, tuple):
                return float(value[0]) / value[1] if value[1] != 0 else 0
            elif isinstance(value, (int, float)):
                return value
            else:
                return float(value)

        focal_length = exif.get("FocalLength")
        focal_length_35mm = exif.get("FocalLengthIn35mmFilm")
        # 필요한 EXIF 데이터가 없으면 예외 발생
        if not focal_length:
            raise ValueError("필요한 EXIF 데이터가 부족합니다.")

        return {
            "focal_length_mm": convert_ifd_rational(focal_length),
            "focal_length_35mm": convert_ifd_rational(focal_length_35mm) if focal_length_35mm else None,
            "image_width": width,
            "image_height": height,
        }
    except Exception as e:
        # EXIF 데이터가 없거나 오류 발생 시 기본값과 이미지 크기 반환
        return {
            "focal_length_mm": 5.4,
            "focal_length_35mm": 23,
            "image_width": width,
            "image_height": height,
        }


def calculate_sensor_size(exif_info):
    focal_length_mm = exif_info["focal_length_mm"]
    focal_length_35mm = exif_info["focal_length_35mm"]
    crop_factor = focal_length_35mm / focal_length_mm
    diagonal_35mm = (36**2 + 24**2)**0.5
    sensor_diagonal = diagonal_35mm / crop_factor
    aspect_ratio_width = 4
    aspect_ratio_height = 3
    sensor_width = sensor_diagonal * (aspect_ratio_width / ((aspect_ratio_width**2 + aspect_ratio_height**2)**0.5))
    sensor_height = sensor_diagonal * (aspect_ratio_height / ((aspect_ratio_width**2 + aspect_ratio_height**2)**0.5))
    return sensor_width, sensor_height

def save_image_with_exif(input_image, save_path):
    exif_data = input_image.info.get("exif")
    if exif_data:
        input_image.save(save_path, exif=exif_data)
    else:
        input_image.save(save_path)

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
def inference(input_image, text_prompt, ref_class_name, ref_width_mm, ref_height_mm):
    global last_masked_image, stored_masked_image
    os.makedirs("./upload_image", exist_ok=True)
    save_path = "./upload_image/user_upload.jpg"
    save_image_with_exif(input_image, save_path)

    exif_info = extract_exif(save_path)
    print(f"[DEBUG] EXIF 정보: {exif_info}")

    sensor_width_mm, sensor_height_mm = calculate_sensor_size(exif_info)
    focal_length_mm = exif_info["focal_length_mm"]

    global FOCAL_LENGTH_MM, SENSOR_WIDTH_MM, SENSOR_HEIGHT_MM
    FOCAL_LENGTH_MM = focal_length_mm
    SENSOR_WIDTH_MM = sensor_width_mm
    SENSOR_HEIGHT_MM = sensor_height_mm

    image_source, image_rgb = load_image(save_path)
    sam2_predictor_global.set_image(image_source.copy())

    boxes, confidences, labels = predict(
        model=grounding_model_global,
        image=image_rgb,
        caption=text_prompt,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )

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

    detections = sv.Detections(
        xyxy=input_boxes_xyxy,
        mask=masks.astype(bool),
        class_id=np.arange(len(labels))
    )
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=image_source.copy(), detections=detections)
    label_annotator = sv.LabelAnnotator()
    confidences_np = confidences.numpy() if isinstance(confidences, torch.Tensor) else confidences
    labels_text = [f"{cls_name} {conf:.2f}" for cls_name, conf in zip(labels, confidences_np)]
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels_text)
    mask_annotator = sv.MaskAnnotator()
    annotated_frame_with_mask = mask_annotator.annotate(scene=annotated_frame.copy(), detections=detections)

    masked_image_pil = Image.fromarray(annotated_frame_with_mask)
    last_masked_image = masked_image_pil
    stored_masked_image = masked_image_pil  # Masked Image를 지속적으로 저장
    os.makedirs("./upload_image", exist_ok=True)
    masked_image_pil.save("./upload_image/masked_image.jpg")

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
        prompt_style = (
            f"Describe the style of the {cls_name}. Is it modern, minimal, natural, vintage, classic, French, Nordic, industrial, lovely, Korean, or unique?"
        )

        input_style = blip2_processor_global(images=cropped_img, text=prompt_style, return_tensors="pt").to(DEVICE)
        output_style = blip2_model_global.generate(**input_style)
        style_description = blip2_processor_global.decode(output_style[0], skip_special_tokens=True)
        ann["style"] = restrict_style(style_description)

    depth_map = generate_depth_map(image_source, midas_model_global, midas_processor_global)
    depth_min, depth_max = depth_map.min(), depth_map.max()
    depth_map_norm = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)
    depth_map_vis = (depth_map_norm * 255).astype(np.uint8)
    depth_map_pil = Image.fromarray(depth_map_vis)
    
    relative_depth_map = generate_relative_depth_map(image_source, midas_model_global, midas_processor_global)
    h_img, w_img, _ = image_source.shape

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

    for ann in annotations:
        if ann["class_name"].lower() == ref_class_name.lower():
            continue
        obj_mask = get_mask(ann["segmentation"])
        obj_px_w, obj_px_h = measure_2d_size_from_mask(obj_mask)
        obj_avg_depth = get_average_relative_depth(real_depth_map, obj_mask)

        real_width = (obj_px_w * obj_avg_depth * SENSOR_WIDTH_MM) / (FOCAL_LENGTH_MM * w_img)
        real_height = (obj_px_h * obj_avg_depth * SENSOR_HEIGHT_MM) / (FOCAL_LENGTH_MM * h_img)+300

        ann["width"] = round(real_width, 2)
        ann["height"] = round(real_height, 2)
    
    results_for_table = []
    for ann in annotations:
        width = ann.get("width", 0.0)
        height = ann.get("height", 0.0)
        results_for_table.append([
            ann["class_name"],
            ann.get("style", ""),
            width,
            height
        ])

    results_json_path = "./upload_image/results.json"
    with open(results_json_path, "w") as f:
        json.dump(results_for_table, f, ensure_ascii=False, indent=2)

    return masked_image_pil, depth_map_pil, results_for_table, json.dumps(exif_info, indent=2)

def restrict_style(style_text):
    allowed_styles = ["modern", "natural", "lovely","minimal","vintage","classic","French","Nordic","industrial","korean","unique"]
    style_text_lower = style_text.lower()
    for allowed in allowed_styles:
        if allowed in style_text_lower:
            return allowed
    # 만약 허용된 스타일 중 하나도 발견되지 않으면, 기본값 또는 "unknown" 반환
    return "unknown"


def recommend_page(text_prompt, input_image, user_prompt):
    rec_recommendations = recommend_interface(user_prompt)
    global stored_masked_image
    return stored_masked_image, user_prompt, rec_recommendations






# --------------------------------------------------------------
# 6. Gradio 인터페이스 (탭 사용)
# --------------------------------------------------------------
def create_demo():
    # 인터페이스 생성 전에 초기 Masked Image 불러오기
    initial_masked_image = None
    masked_image_path = "./upload_image/masked_image.jpg"
    if os.path.exists(masked_image_path):
        initial_masked_image = Image.open(masked_image_path)
    
    with gr.Blocks() as demo:
        gr.Markdown("## 오늘의 바뀔 내 집 DEMO")
        with gr.Tabs():
            with gr.TabItem("추론"):
                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(label="Upload Image", type="pil")
                        text_prompt = gr.Textbox(label="Text Prompt", value="bed. chair. table. curtains. mirror.")
                        ref_class_name = gr.Textbox(label="Reference Class Name", value="chair")
                        ref_width_mm = gr.Number(label="Reference Width (mm)", value=420.0)
                        ref_height_mm = gr.Number(label="Reference Height (mm)", value=800.0)
                        run_button = gr.Button("Run Inference")
                    with gr.Column():
                        masked_image_out = gr.Image(label="Masked Image")
                        depth_map_out = gr.Image(label="Depth Map")
                        results_table_out = gr.Dataframe(
                            headers=["Class", "Style", "Width(mm)", "Height(mm)"],
                            datatype=["str", "str", "number", "number"],
                            label="Results Table"
                        )
                        exif_info_text = gr.Textbox(label="EXIF Information", interactive=False, lines=10)
                run_button.click(
                    fn=inference,
                    inputs=[input_image, text_prompt, ref_class_name, ref_width_mm, ref_height_mm],
                    outputs=[masked_image_out, depth_map_out, results_table_out, exif_info_text]
                )
            with gr.TabItem("가구 추천"):
                with gr.Row():
                    with gr.Column():
                        user_prompt = gr.Textbox(label="User Prompt", placeholder="스타일을 입력하세요...")
                        rec_masked_image = gr.Image(value=initial_masked_image, label="Masked Image", height=400)
                        rec_out_recommendations = gr.Markdown(label="추천 결과")
                recommend_button = gr.Button("가구 추천")
                recommend_button.click(
                    fn=recommend_page,
                    inputs=[text_prompt, input_image, user_prompt],
                    outputs=[rec_masked_image, user_prompt, rec_out_recommendations]
                )

    return demo

if __name__ == "__main__":
    demo_app = create_demo()
    demo_app.launch()