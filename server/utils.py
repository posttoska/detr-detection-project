import io
import torch
import torchvision.transforms as transforms
from model.detr import DETR
from config.config import config
from PIL import Image

num_classes = 23
bg_class_idx = 0

CLASSES = [
    "background","person","bird","cat","cow","dog","horse","sheep","aeroplane","bicycle",
    "boat","bus","car","motorbike","train","bottle","chair","diningtable","pottedplant",
    "sofa","tvmonitor","apple","cup"
]

# model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DETR(config, num_classes=num_classes, bg_class_idx=bg_class_idx).to(device)

state = torch.load(r'C:\Users\posttoska\Documents\detr-detection-project\voc\detr_voc23cls_plus_mydata.pth', map_location=device)
# load trained weights
model.load_state_dict(state)
model.eval()

# inference
def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize((640, 640)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])

    # convert to RGB and keep original size
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    W, H = image.size
    return my_transforms(image).unsqueeze(0), (W, H)

# get prediction from model
@torch.no_grad()
def get_prediction(image_bytes: bytes, score_thresh=0.5, use_nms=True):
    x, (W, H) = transform_image(image_bytes)
    x = x.to(device)

    out = model(x, score_thresh=score_thresh, use_nms=use_nms)
    det = out['detections'][0]              # list element for this image

    boxes  = det['boxes'].cpu()             # normalized xyxy in [0,1], shape (N,4)
    scores = det['scores'].cpu()            # (N,)
    labels = det['labels'].cpu()            # (N,)

    # convert to pixel coords of the ORIGINAL image
    boxes_px = boxes.clone()
    boxes_px[:, [0, 2]] *= W
    boxes_px[:, [1, 3]] *= H

    return boxes_px, scores, labels

def _to_list(t: torch.Tensor):
    return t.detach().cpu().tolist()

def get_result(image_file):
    image_bytes = image_file.file.read()
    boxes, scores, labels = get_prediction(image_bytes)
    print(boxes, scores, labels)
    label_ids = _to_list(labels)
    result = {
        "message": "Hello from predict",
        "predictions": {
            "boxes": _to_list(boxes),
            "scores": _to_list(scores),
            "labels": label_ids,
            "label_names": [CLASSES[i] for i in label_ids],
        }
    }
    return result
