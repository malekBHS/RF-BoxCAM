import argparse
import math
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

class RFBoxCAM:
    def __init__(self, weights_path, layer_idx=17, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(weights_path).model.to(self.device)
        self.layer_idx = layer_idx

    @staticmethod
    def iou(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        x1_min, y1_min = x1 - w1 / 2, y1 - h1 / 2
        x1_max, y1_max = x1 + w1 / 2, y1 + h1 / 2

        x2_min, y2_min = x2 - w2 / 2, y2 - h2 / 2
        x2_max, y2_max = x2 + w2 / 2, y2 + h2 / 2

        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
        union_area = (w1 * h1) + (w2 * h2) - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0

    def get_overlapping_boxes(self, results, target_idx, conf_thresh=0.25, iou_thresh=0.45):
        xt, yt, wt, ht = results[0:4, target_idx]
        target_box = [xt, yt, wt, ht]
        
        target_class = np.argmax(results[4:, target_idx])
        valid_boxes = {}
        
        for i in range(results.shape[1]):
            scores = results[4:, i]
            pred_class = np.argmax(scores)
            conf = scores[pred_class]
            
            if conf < conf_thresh or pred_class != target_class:
                continue
                
            if self.iou(results[0:4, i], target_box) > iou_thresh:
                valid_boxes[i] = conf

        if not valid_boxes:
            return {}
            
        conf_sum = sum(math.exp(c) for c in valid_boxes.values())
        return {k: math.exp(v) / conf_sum for k, v in valid_boxes.items()}

    def __call__(self, img_path, target_idx):
        img = cv2.imread(img_path)
        img_resized = cv2.resize(img, (640, 640))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        tensor_img = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor_img = tensor_img.to(self.device)
        tensor_img.requires_grad_(True)

        with torch.no_grad():
            raw_preds = self.model(tensor_img)[0][0].cpu().numpy()
        
        weights = self.get_overlapping_boxes(raw_preds, target_idx)
        final_hm = np.zeros((640, 640))

        acts, grads = {}, {}

        def fwd_hook(m, i, o):
            acts['val'] = o

        def bwd_hook(m, gi, go):
            grads['val'] = go[0]

        layer = self.model.model[self.layer_idx]
        layer.requires_grad_(True)
        
        h_fwd = layer.register_forward_hook(fwd_hook)
        h_bwd = layer.register_full_backward_hook(bwd_hook)

        out = self.model(tensor_img)

        for box_idx, w in weights.items():
            self.model.zero_grad(set_to_none=True)
            if tensor_img.grad is not None:
                tensor_img.grad.zero_()

            # dynamically grab the confidence score for the specific predicted class
            target_class = np.argmax(raw_preds[4:, box_idx])
            score = out[0][0][4 + target_class][box_idx]
            score.backward(retain_graph=True)

            act = acts['val']
            grad = grads['val']

            weighted_acts = grad[0] * act
            hm = torch.relu(torch.sum(weighted_acts[0], dim=0))
            
            box_hm = np.zeros((640, 640))
            active_cells = torch.nonzero(hm).cpu().numpy()
            
            for y, x in active_cells:
                self.model.zero_grad(set_to_none=True)
                if tensor_img.grad is not None:
                    tensor_img.grad.zero_()
                    
                hm[y, x].backward(retain_graph=True)
                
                in_grad = tensor_img.grad.detach().cpu().numpy()[0]
                gradmap = np.linalg.norm(np.transpose(in_grad, (1, 2, 0)), axis=2)
                
                smooth_rf = cv2.GaussianBlur(gradmap, (21, 21), 0)
                if np.max(smooth_rf) > np.min(smooth_rf):
                    smooth_rf = (smooth_rf - np.min(smooth_rf)) / (np.max(smooth_rf) - np.min(smooth_rf))
                
                box_hm += hm[y, x].item() * smooth_rf

            if np.max(box_hm) > np.min(box_hm):
                box_hm = (box_hm - np.min(box_hm)) / (np.max(box_hm) - np.min(box_hm))

            final_hm += w * box_hm

        h_fwd.remove()
        h_bwd.remove()

        if np.max(final_hm) > np.min(final_hm):
            final_hm = (final_hm - np.min(final_hm)) / (np.max(final_hm) - np.min(final_hm))

        return img_rgb, final_hm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--out", type=str, default="output.jpg")
    
    args = parser.parse_args()

    explainer = RFBoxCAM(args.weights)
    img, hm = explainer(args.image, args.target)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.imshow(hm, cmap='jet', alpha=0.4)
    plt.axis("off")
    plt.savefig(args.out, bbox_inches="tight", pad_inches=0)
    plt.close()