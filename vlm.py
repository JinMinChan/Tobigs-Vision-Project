from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import json

# BLIP-2 모델 로드
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl")

# JSON 파일 로드
json_path = "./result/results.json"
with open(json_path, "r") as file:
    data = json.load(file)

image_path = data["image_path"]
annotations = data["annotations"]

# 이미지 로드
image = Image.open(image_path).convert("RGB")

# 바운딩 박스를 사용하여 객체 영역 크롭
def crop_object(image, bbox):
    x_min, y_min, x_max, y_max = bbox
    return image.crop((x_min, y_min, x_max, y_max))

# 스타일 카테고리
def determine_style(description):
    styles = [
        "Modern", "Minimal", "Natural", "Vintage", "Classic", "French",
        "Nordic", "Industrial", "Lovely", "Korean", "Unique"
    ]
    matched_styles = [style for style in styles if style.lower() in description.lower()]
    return matched_styles[0] if matched_styles else "Unknown"

# 색상 카테고리
def determine_color(description):
    colors = [
        "White", "Black", "Gray", "Red", "Blue", "Green", "Yellow",
        "Brown", "Beige", "Pink", "Purple", "Orange"
    ]
    matched_colors = [color for color in colors if color.lower() in description.lower()]
    return matched_colors[0] if matched_colors else "Unknown"

# 결과 저장을 위한 리스트
results = []

# 각 객체에 대해 색상, 스타일 추출
for annotation in annotations:
    class_name = annotation["class_name"]
    bbox = annotation["bbox"]

    # 객체 크롭
    cropped_image = crop_object(image, bbox)

    # 색상 요청
    text_input_color = f"Describe the color of the {class_name}. Is it white, black, gray, red, blue, green, yellow, brown, beige, pink, purple, or orange?"
    inputs_color = processor(images=cropped_image, text=text_input_color, return_tensors="pt")
    output_color = model.generate(**inputs_color)
    description_color = processor.decode(output_color[0], skip_special_tokens=True)
    color = determine_color(description_color)

    # 스타일 요청
    text_input_style = f"Describe the style of the {class_name}. Is it modern, minimal, natural, vintage, classic, French, Nordic, industrial, lovely, Korean, or unique?"
    inputs_style = processor(images=cropped_image, text=text_input_style, return_tensors="pt")
    output_style = model.generate(**inputs_style)
    description_style = processor.decode(output_style[0], skip_special_tokens=True)
    style = determine_style(description_style)

    # 결과 저장
    results.append({
        "class_name": class_name,
        "color": color,
        "style": style
    })

# 결과 출력
for result in results:
    print(f"Class: {result['class_name']}")
    print(f"Color: {result['color']}")
    print(f"Style: {result['style']}\n")

# 결과를 JSON 파일로 저장
output_path = "./result/furniture_descriptions.json"
with open(output_path, "w") as file:
    json.dump(results, file, indent=4)

print(f"Descriptions saved to {output_path}")
