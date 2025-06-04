import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from torchvision.io import read_video

# 모델 로딩
model = models.resnet18(pretrained=True)
model.eval()

# ImageNet 클래스 이름 로딩
with open("imagenet_classes.txt") as f:
    classes = [line.strip() for line in f.readlines()]

# 추론 함수
def run_vision_model(mp4_path):
    video_frames, _, _ = read_video(mp4_path)
    first_frame = video_frames[0].permute(2, 0, 1)  # (HWC → CHW)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(first_frame / 255.0).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.nn.functional.softmax(output[0], dim=0)
        conf, pred_idx = torch.max(prob, dim=0)

    label = classes[pred_idx]
    return f"🌐 추론 결과: {label} ({conf.item():.2f})"
