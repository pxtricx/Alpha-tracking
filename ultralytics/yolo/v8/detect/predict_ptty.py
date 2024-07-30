import os
import hydra
import torch
import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque
import numpy as np
from skimage.metrics import structural_similarity as ssim  # for similarity measure

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}

largest_box_dict = {} 
deepsort = None

unique_id = 0
existing_ids = set()
object_images = {}  # Dictionary to store the latest image for each identity

def init_tracker():
    global deepsort
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                        max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    if label == 0: #person
        color = (85,45,255)
    
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)
    
    cv2.circle(img, (x1 +r, y1+r), 2, color, 12)
    cv2.circle(img, (x2 -r, y1+r), 2, color, 12)
    cv2.circle(img, (x1 +r, y2-r), 2, color, 12)
    cv2.circle(img, (x2 -r, y2-r), 2, color, 12)
    
    return img

def UI_box(x, img, color=None, label=None, line_thickness=None, elapsed_time=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

        img = draw_border(img, (c1[0], c1[1] - t_size[1] -3), (c1[0] + t_size[0], c1[1]+3), color, 1, 8, 2)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    
        if elapsed_time:
            label += f' Timer : {elapsed_time}s'
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def draw_boxes(img, bbox, names, object_id, identities=None, offset=(0, 0), object_timings={}):
    global unique_id, existing_ids
    height, width, _ = img.shape
    for key in list(data_deque):
        if key not in identities:
            data_deque.pop(key)
            existing_ids.discard(key)

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        center = (int((x2 + x1) / 2), int((y2 + y1) / 2))
        id = int(identities[i]) if identities is not None else 0

        if id not in data_deque:
            while unique_id in existing_ids:
                unique_id += 1
            id = unique_id
            data_deque[id] = deque(maxlen=64)
            unique_id += 1

        existing_ids.add(id)
        color = compute_color_for_labels(object_id[i])

        obj_name = names[object_id[i]]
        label = f'{obj_name} {id}'
        object_timings[id] = object_timings.get(id, 0) + 1  # Increment the counter for the object
        elapsed_time = object_timings[id]

        data_deque[id].appendleft(center)
        UI_box(box, img, label=label, color=color, line_thickness=2, elapsed_time=elapsed_time)

        for j in range(1, len(data_deque[id])):
            if data_deque[id][j - 1] is None or data_deque[id][j] is None:
                continue
            cv2.line(img, data_deque[id][j - 1], data_deque[id][j], color, thickness=2)

        # Save the cropped image of the object
        object_image = img[y1:y2, x1:x2]
        object_images[id] = object_image

        # Save the image to the "output" folder
        output_folder = "output"
        os.makedirs(output_folder, exist_ok=True)
        output_image_path = os.path.join(output_folder, f"{id}.jpg")
        cv2.imwrite(output_image_path, object_image)

    return img

def plot_boxes_on_image(image, boxes, object_id, class_names, identities, object_timings):
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        object_id = object_id[i]
        identity = identities[i]
        elapsed_time = object_timings.get(identity, 0)

        label = f'{class_names[object_id]} {identity} {elapsed_time}s'
        color = compute_color_for_labels(object_id)
        image = UI_box((x1, y1, x2, y2), image, label=label, color=color, line_thickness=2)

    return image

@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def run(cfg):
    # Argument parsing
    parser = argparse.ArgumentParser(description="Object Tracking and Counting")
    parser.add_argument('--model', default='yolov8l.pt', help='Path to the model')
    parser.add_argument('--source', default='test4.mp4', help='Source video file or directory')
    parser.add_argument('--img-size', type=int, default=640, help='Image size for inference')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold for detections')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='Device to run inference on (e.g., "cpu" or "cuda:0")')
    parser.add_argument('--view-img', action='store_true', help='Display the results')
    parser.add_argument('--save-txt', action='store_true', help='Save results to *.txt files')
    parser.add_argument('--save-conf', action='store_true', help='Save confidences in output labels')
    parser.add_argument('--save-crop', action='store_true', help='Save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='Do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='Filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='Class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='Augmented inference')
    parser.add_argument('--update', action='store_true', help='Update all models')
    parser.add_argument('--project', default='runs/detect', help='Save results to project/name')
    parser.add_argument('--name', default='exp', help='Save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='Existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='Bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='Hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='Hide confidences')
    parser.add_argument('--half', action='store_true', help='Use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='Use OpenCV DNN for ONNX inference')
    parser.add_argument('--show', type=bool, default=False, help='Display the inference results')
    opt = parser.parse_args()
    
    source = opt.source
    show = opt.show

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).autoshape()  
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_imgsz(opt.img_size, s=stride)  

    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs = 1
    vid_path, vid_writer = None, None

    if device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # run once

    old_img_w = old_img_h = imgsz
    old_img_b = 1

    init_tracker()
    object_timings = {}  # Dictionary to store object timings

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()  
        img /= 255.0  
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t1 = time.time()
        pred = model(img, augment=opt.augment)[0]
        t2 = time.time()

        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms)
        for i, det in enumerate(pred):
            p, s, im0, frame = path, '', im0s.copy(), dataset.frame
            p = Path(p)  

            save_path = str(Path(opt.project) / opt.name / p.name)  
            txt_path = str(Path(opt.project) / opt.name / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
            s += '%gx%g ' % img.shape[2:]  

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                xywhs = xyxy_to_xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    object_id = outputs[:, -2]

                    draw_boxes(im0, bbox_xyxy, names, object_id, identities, object_timings=object_timings)

                for *xyxy, conf, cls in reversed(det):
                    if opt.save_txt:  
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh, conf))

                    if opt.save_img or opt.save_crop or opt.show:
                        c = int(cls)  
                        label = f'{names[c]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=opt.line_thickness)
                        if opt.save_crop:
                            save_one_box(xyxy, im0, file=Path(opt.project) / opt.name / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            if opt.save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  
                        fourcc = 'mp4v'  
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

            if opt.view_img:
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  
                    raise StopIteration

        print(f'Done. ({t2 - t1:.3f}s)')

    if opt.save_txt or opt.save_img:
        print(f"Results saved to {Path(opt.project) / opt.name}")
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    if opt.update:  
        strip_optimizer(Path(opt.weights))
