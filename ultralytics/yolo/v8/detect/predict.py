import os
import shutil
import sys
current_file_path = os.path.abspath(os.path.dirname(__file__))
deep_sort_path = os.path.abspath(os.path.join(current_file_path, 'deep_sort_pytorch'))


project_root = os.path.abspath(os.path.join(current_file_path, '..', '..', '..', '..'))
deep_sort_path = os.path.join(project_root, 'ultralytics', 'yolo', 'v8', 'detect', 'deep_sort_pytorch')

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if deep_sort_path not in sys.path:
    sys.path.insert(0, deep_sort_path)

sys.path.append(os.path.dirname(__file__))

import hydra
from matplotlib import pyplot as plt
import pandas as pd
import torch
import argparse
import time
from pathlib import Path
from PIL import Image
import cv2
import torch
import torch.backends.cudnn as cudnn
from torchvision import models, transforms
from numpy import random
from ultralytics.yolo.engine import predictor
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.ops import xyxy2xywh
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from skimage.metrics import structural_similarity as ssim
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from scipy.spatial.distance import cosine
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image

current_file_path = os.path.abspath(os.path.dirname(__file__))
deep_sort_path = os.path.abspath(os.path.join(current_file_path, 'deep_sort_pytorch'))
if deep_sort_path not in sys.path:
    sys.path.append(deep_sort_path)

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}

largest_box_dict = {}
deepsort = None

unique_id = 0
existing_ids = set()

model = ResNet50(weights='imagenet', include_top=False)

def init_tracker():
    global deepsort
    cfg_deep = get_config()
    config_file = os.path.join(os.path.dirname(__file__), "deep_sort_pytorch", "configs", "deep_sort.yaml")
    print(f"Attempting to open config file at: {config_file}")
    if os.path.exists(config_file):
        cfg_deep.merge_from_file(config_file)
    else:
        raise FileNotFoundError(f"Config file not found at {config_file}")
    
    print("Configuration contents:")
    print(cfg_deep)

    # Update the model path to use an absolute path
    reid_ckpt = os.path.join(os.path.dirname(__file__), cfg_deep.DEEPSORT.REID_CKPT)
    print(f"ReID checkpoint path: {reid_ckpt}")

    if not os.path.exists(reid_ckpt):
        raise FileNotFoundError(f"ReID checkpoint not found at {reid_ckpt}")

    try:
        deepsort = DeepSort(reid_ckpt,
                            max_dist=cfg_deep.DEEPSORT.MAX_DIST,
                            min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
                            max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg_deep.DEEPSORT.MAX_AGE,
                            n_init=cfg_deep.DEEPSORT.N_INIT,
                            nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                            use_cuda=True)
        print("DeepSort initialized successfully")
    except Exception as e:
        print(f"Error initializing DeepSort: {e}")
        raise

    return deepsort, cfg_deep
##########################################################################################
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

        center = (int((x2+x1)/2), int((y2+y1)/2))
        id = int(identities[i]) if identities is not None else 0

        if id not in data_deque:
            while unique_id in existing_ids:
                unique_id += 1
            id = unique_id
            unique_id += 1
            data_deque[id] = deque(maxlen=64)
            existing_ids.add(id)

        if id not in object_timings:
            object_timings[id] = time.time()

        elapsed_time = time.time() - object_timings[id]
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

        color = compute_color_for_labels(object_id[i])
        obj_name = names[object_id[i]]
        label = '{}{:d}'.format("", id) + ":" + '%s' % (obj_name)

        data_deque[id].appendleft(center)
        UI_box(box, img, label=label, color=color, line_thickness=2, elapsed_time=elapsed_time)

        for i in range(1, len(data_deque[id])):
            if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                continue
            thickness = int(np.sqrt(64 / float(i + i)) * 1.5)
            cv2.line(img, data_deque[id][i - 1], data_deque[id][i], color, thickness)

    return img

def analyze_saved_images(save_dir):
    # Load pre-trained ResNet50 model
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.eval()
    
    # Define image transformations
    preprocess = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def extract_features(image_path):
        image = Image.open(image_path).convert('RGB')
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)
        with torch.no_grad():
            features = model(input_batch)
        return features.squeeze().numpy()

    results = []

    # Get all folder names that start with "obj_id_" and sort them
    obj_folders = sorted([folder for folder in os.listdir(save_dir) if folder.startswith("obj_id_")],
                         key=lambda x: int(x.split('_')[2]))  # Sort based on the numeric ID


    for obj_folder in obj_folders:
        folder_path = os.path.join(save_dir, obj_folder)
        image_files = sorted(os.listdir(folder_path))
        
        if len(image_files) < 2:
            print(f"Not enough images in {obj_folder} for comparison")
            results.append({
                'Folder': obj_folder,
                'Average Similarity': None,
                'Number of Images': len(image_files),
                'Conclusion': 'Not enough images for comparison'
            })
            continue
        
        reference_features = extract_features(os.path.join(folder_path, image_files[0]))
        
        similarities = []
        for image_file in image_files[1:]:
            image_path = os.path.join(folder_path, image_file)
            features = extract_features(image_path)
            similarity = 1 - cosine(reference_features, features)
            similarities.append(similarity)
        
        avg_similarity = sum(similarities) / len(similarities)
        
        conclusion = "Images likely contain the same object" if avg_similarity > 0.7 else "Images may contain different objects"
        
        results.append({
            'Folder': obj_folder,
            'Average Similarity': avg_similarity,
            'Number of Images': len(image_files),
            'Conclusion': conclusion
        })
        
        print(f"Folder {obj_folder}:")
        print(f"  Average similarity: {avg_similarity:.4f}")
        print(f"  Number of images: {len(image_files)}")
        print(f"  Conclusion: {conclusion}")
        print()

    # Create a DataFrame from the results
    df_images = pd.DataFrame(results)

    return df_images

def compute_similarity(img1, img2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Resize images to have the same dimensions
    height = min(gray1.shape[0], gray2.shape[0])
    width = min(gray1.shape[1], gray2.shape[1])
    gray1 = cv2.resize(gray1, (width, height))
    gray2 = cv2.resize(gray2, (width, height))
    
    # Compute SSIM between two images
    score, _ = ssim(gray1, gray2, full=True)
    return score

def get_image_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    return features.flatten()

def compare_images(img1_path, img2_path):
    feat1 = get_image_features(img1_path)
    feat2 = get_image_features(img2_path)
    similarity = 1 - cosine(feat1, feat2)
    return similarity

def create_person_count_df(largest_box_dict):
    df = pd.DataFrame(list(largest_box_dict.items()), columns=['Frame', 'Person_Count'])
    return df

class DetectionPredictor(BasePredictor):
    
    def __init__(self, cfg, cls=[0], names="Person"): # เปลี่ยน class ตรงนี้ เพราะแก้ให้มันรัน terminal ไม่เป็น
        super().__init__(cfg)
        self.names = names or [] # Store the names separately
        self.cls = cls
        self.object_timings = {}
        self.ground_truths = []
        self.predictions = []
        self.person_class_id = 0
        self.last_images = {}  # Initialize the last_images dictionary
        self.similarity_threshold = 0.5  # similarity threshold (0.8 high level of similarity)
        self.last_detections = {}
        self.output_video_path = None

    def compute_similarity(self, img1, img2):
        # Convert images to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Resize images to have the same dimensions
        height = min(gray1.shape[0], gray2.shape[0])
        width = min(gray1.shape[1], gray2.shape[1])
        gray1 = cv2.resize(gray1, (width, height))
        gray2 = cv2.resize(gray2, (width, height))
        
        # Compute SSIM between two images
        score, _ = ssim(gray1, gray2, full=True)
        return score

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img
  

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds
    
    
    def write_results(self, idx, preds, batch):
        global deepsort
        p, im, im0 = batch
        all_outputs = []
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]
        self.seen += 1    
        im0 = im0.copy()
        if self.webcam:
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        save_path = str(self.save_dir / p.name)
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        all_outputs.append(det)
        
        if len(det) == 0:
            return log_string
        
        # all initial detections are potential persons
        self.ground_truths.extend([1] * len(det))

        # Filter detections to keep only persons
        mask = torch.tensor([c in self.cls for c in det[:, 5]], device=det.device)
        det = det[mask]


        # Add predictions based on the person filter
        self.predictions.extend([1 if m else 0 for m in mask])

        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
        
        
        xywhs = xyxy2xywh(det[:, 0:4])
        confs = det[:, 4]
        clss = det[:, 5]
        
        mask = clss == self.person_class_id
        xywhs = xywhs[mask]
        confs = confs[mask]
        clss = clss[mask]

        # Log the detections before updating DeepSort
        print(f"Frame {frame}: Detections before DeepSort - {len(xywhs)}")

        outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
        
        # Log the number of outputs from DeepSort
        print(f"Frame {frame}: Outputs from DeepSort - {len(outputs)}")

        person_count = len(outputs)
        largest_box_dict[frame] = person_count

        '''
        person_count = 0
        for *xyxy, conf, cls in reversed(det):
            if cls == self.person_class_id:
                person_count += 1
        largest_box_dict[frame] = person_count
        '''

        if len(outputs) > 0:
            for j, (output, conf) in enumerate(zip(outputs, confs)):
                bboxes = output[0:4]
                id = output[4]
                cls = output[5]

                c = int(cls)
                label = f'{id} {self.names[c] if c < len(self.names) else "Unknown"} {conf:.2f}'
                self.annotator.box_label(bboxes, label, color=colors(c, True))

                x1, y1, x2, y2 = map(int, bboxes)
                crop_img = im0[y1:y2, x1:x2]

                if id not in self.last_detections or frame - self.last_detections[id] >= 10:
                    obj_folder = self.save_dir / f"obj_id_{id}"
                    obj_folder.mkdir(parents=True, exist_ok=True)

                    frame_filename = f"{id}_{frame}.jpg"
                    cv2.imwrite(str(obj_folder / frame_filename), crop_img)
                    self.last_detections[id] = frame

                self.last_images[id] = crop_img

                obj_folder = self.save_dir / f"obj_id_{id}"
                obj_folder.mkdir(parents=True, exist_ok=True)

                frame_filename = f"{id}_{frame}.jpg"
                img_path = str(obj_folder / frame_filename)
                #cv2.imwrite(img_path, crop_img)
                cv2.imwrite(str(obj_folder / frame_filename), crop_img)

                self.ground_truths.append(0)  # Assuming all are persons (class 0)
                self.predictions.append(cls)

        return log_string

    def calculate_metrics(self):
        
        cm = confusion_matrix(self.ground_truths, self.predictions)
        precision = precision_score(self.ground_truths, self.predictions, average='binary')
        recall = recall_score(self.ground_truths, self.predictions, average='binary')
        f1 = f1_score(self.ground_truths, self.predictions, average='binary')
        accuracy = accuracy_score(self.ground_truths, self.predictions)

        print(f"Confusion Matrix:\n{cm}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"Accuracy: {accuracy:.4f}")

        # Create a DataFrame with the metrics
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy','Precision', 'Recall', 'F1-score'],
            'Value': [accuracy, precision, recall, f1]
        })

        # Create a DataFrame for the confusion matrix
        cm_df = pd.DataFrame(cm, columns=['Predicted Negative', 'Predicted Positive'], 
                            index=['Actual Negative', 'Actual Positive'])

        return metrics_df, cm_df  # Return the DataFrames

    def calculate_mean_person_count(self):
        if largest_box_dict:
            mean_person_count = int(np.mean(list(largest_box_dict.values())))
            max_person_count = int(np.max(list(largest_box_dict.values())))
            min_person_count = int(np.min(list(largest_box_dict.values())))
            print(f"Mean person count: {mean_person_count}")
            person_data_df = pd.DataFrame({'Mean': ['Person Mean','Person Max', 'Person Min'],
                                              'Value': [mean_person_count, max_person_count, min_person_count]})
            return person_data_df
        else:
            print("No person count data available.")
            return 0.0


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    start_time = time.time()
    init_tracker()
    cfg.model = cfg.model or "yolov8l.pt"
    print(cfg.model)
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    cfg.vid_stride = 2  # Process every 2 frame
    predictor = DetectionPredictor(cfg, cls=[0]) # Only detect persons (class 0)
    predictor()


    # Analyze saved images
    df_images = analyze_saved_images(predictor.save_dir)
    # Calculate metrics
    metrics_df, cm_df = predictor.calculate_metrics()
    # Create DataFrame for person count
    person_count_df = create_person_count_df(largest_box_dict)

    # Calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Create DataFrame for processing time
    time_df = pd.DataFrame({
        'Metric': ['Processing Time (seconds)'],
        'Value': [elapsed_time]
    })
    person_data_df = predictor.calculate_mean_person_count()

    results_dir = 'D:/VScode/a project-thesis/Alpha-tracking/video_prediction/media/results'
    excel_path = os.path.join(results_dir, 'detection_and_analysis_results_yolov8x.xlsx')

    with pd.ExcelWriter(excel_path) as writer:
        df_images.to_excel(writer, sheet_name='Image Analysis', index=False)
        metrics_df.to_excel(writer, sheet_name='Detection Metrics', index=False)
        cm_df.to_excel(writer, sheet_name='Confusion Matrix')
        person_count_df.to_excel(writer, sheet_name='Person Count per Frame', index=False)
        person_data_df.to_excel(writer, sheet_name='Person Count mean', index=False)
        time_df.to_excel(writer, sheet_name='Processing Time', index=False)
    print(f"All results saved to {excel_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(person_count_df['Frame'], person_count_df['Person_Count'], marker='o', linestyle='-')
    plt.xlabel('Frame')
    plt.ylabel('Person Count')
    plt.title('Number of Persons Detected per Frame')
    plot_path = os.path.join(predictor.save_dir, 'person_count_plot.png')
    plt.savefig(plot_path)
    print(f"Person count plot saved to {plot_path}")

       
    end_time = time.time()
    elapsed_time = end_time - start_time
    


if __name__ == "__main__":
    start_time = time.time()
    predict()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"time: {elapsed_time:.2f} sec")
    print(f"person count {largest_box_dict}") # count person : frame by frame

