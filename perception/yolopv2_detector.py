import torch
import numpy as np
import cv2
from perception.utils import non_max_suppression, scale_coords, split_for_trace_model, letterbox

class YOLOPv2Detector:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        self.img_size = (640, 640) # Standard YOLOPv2 inference size
        # Letterbox state — set during preprocess, used during postprocess
        self._letterbox_shape = (384, 640)  # default (H, W)
        self._ratio = (1.0, 1.0)
        self._pad = (0.0, 0.0)

    def preprocess(self, img):
        # Image is (H, W, C) BGR (numpy)
        # Use official letterbox for best results — preserves aspect ratio
        img_lb, ratio, pad = letterbox(img, self.img_size, stride=32)
        self._letterbox_shape = img_lb.shape[:2]  # (H, W) of the letterboxed image
        self._ratio = ratio
        self._pad = pad

        # BGR to RGB, (C, H, W)
        img_lb = img_lb[:, :, ::-1].transpose(2, 0, 1)
        img_lb = np.ascontiguousarray(img_lb)
        
        img_tensor = torch.from_numpy(img_lb).to(self.device).float()
        img_tensor /= 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        return img_tensor

    def postprocess(self, out, img0_shape, conf_thres=0.3, iou_thres=0.45):
        # out[0]: detections (tuple: (list of 3 tensors, tuple of 3 tensors)) 
        pred = split_for_trace_model(out[0][0], out[0][1])
        
        # Apply NMS
        detections = non_max_suppression(pred, conf_thres, iou_thres)[0]
        
        # Rescale from letterboxed inference size back to original image size
        if len(detections):
            # Use actual letterboxed shape (H, W), NOT self.img_size which is target (W, H)
            lb_h, lb_w = self._letterbox_shape
            detections[:, :4] = scale_coords(
                (lb_h, lb_w), detections[:, :4], img0_shape
            ).round()
            detections = detections.cpu().numpy().tolist()
        else:
            detections = []

        # Segmentation using official utility functions
        from perception.utils import driving_area_mask, lane_line_mask
        drivable = driving_area_mask(out[1])
        lanes = lane_line_mask(out[2])
        
        return drivable, lanes, detections

    def infer(self, img):
        img_tensor = self.preprocess(img)
        with torch.no_grad():
            out = self.model(img_tensor)
        return self.postprocess(out, img.shape)

    def visualize(self, img, drivable, lanes, detections=None):
        h, w = img.shape[:2]
        
        # Official masks are often int32 and might be at a different resolution (e.g. 1280x720)
        # Convert to uint8 for OpenCV compatibility
        drivable_uint8 = drivable.astype(np.uint8)
        lanes_uint8 = lanes.astype(np.uint8)

        drivable_res = cv2.resize(drivable_uint8, (w, h))
        lanes_res = cv2.resize(lanes_uint8, (w, h))
        
        # Create masks
        drivable_mask = np.zeros_like(img)
        drivable_mask[drivable_res > 0] = [0, 255, 0] # Green for drivable (mask is 0 or 1)
        
        lanes_mask = np.zeros_like(img)
        lanes_mask[lanes_res > 0] = [0, 0, 255] # Red for lanes (mask is 0 or 1)
        
        # Overlay
        alpha = 0.5
        im_vis = cv2.addWeighted(img, 1.0, drivable_mask, alpha, 0)
        im_vis = cv2.addWeighted(im_vis, 1.0, lanes_mask, alpha, 0)
        
        # Draw detections if any
        if detections:
            for det in detections:
                # det: [x1, y1, x2, y2, conf, cls]
                x1, y1, x2, y2, conf, cls = det
                cv2.rectangle(im_vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        return im_vis
