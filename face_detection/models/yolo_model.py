import torch
from pathlib import Path
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device

class YOLOModel:
    def __init__(self, config_path, weights_path):
        # Load YOLOv5 model
        self.device = select_device('')
        self.model = attempt_load(weights_path, map_location=self.device)
        self.stride = int(self.model.stride.max())  # Model stride
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

    def detect(self, image):
        # Process the image using the YOLOv5 model
        img = torch.from_numpy(image).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = self.model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.5)[0]

        return pred

    def autoshape(self, image_size):
        # Implement autoshape logic if needed
        pass
