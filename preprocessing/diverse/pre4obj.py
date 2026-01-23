import os
import json
import sys

import torch
import torchvision
import numpy as np
from PIL import Image
from tqdm import tqdm
from urllib.parse import unquote
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

class BatchMentionObjectDetector:
    def __init__(self, args):
        self.args = args

        self.faster_rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.faster_rcnn_model.eval()
        if torch.cuda.is_available():
            self.faster_rcnn_model = self.faster_rcnn_model.cuda()

        self.box_threshold = 0.8
        self.mention_obj_top_k = 5
        self.batch_size = 4


        self.output_dir = os.path.dirname(args.data.train_file)

    def detect_objects_batch(self, image_paths):
        images = []
        valid_indices = []

        for i, img_path in enumerate(image_paths):
            try:
                img = Image.open(img_path).convert("RGB")
                images.append(img)
                valid_indices.append(i)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                valid_indices.append(None)

        if not images:
            return [{}] * len(image_paths)

        detection_tensors = []
        for img in images:
            tensor = torch.tensor(np.array(img).transpose(2, 0, 1) / 255.0, dtype=torch.float32)
            if torch.cuda.is_available():
                tensor = tensor.cuda()
            detection_tensors.append(tensor)

        with torch.no_grad():
            detections = self.faster_rcnn_model(detection_tensors)

        all_results = [{}] * len(image_paths)
        for idx, detection in zip(valid_indices, detections):
            if idx is None:
                continue

            boxes = detection['boxes'].cpu().numpy()
            scores = detection['scores'].cpu().numpy()

            detected_objects = []

            if len(scores) > 0:
                sorted_indices = np.argsort(scores)[::-1]
                boxes = boxes[sorted_indices]
                scores = scores[sorted_indices]

                keep = scores >= self.box_threshold
                boxes = boxes[keep]
                scores = scores[keep]

                boxes = boxes[:self.mention_obj_top_k]
                scores = scores[:self.mention_obj_top_k]

                for box, score in zip(boxes, scores):
                    detected_objects.append({
                        'box': box.tolist(),
                        'score': float(score)
                    })

            all_results[idx] = {'boxes': detected_objects}

        return all_results

    def process_mention_dataset(self, input_file, dataset_type):
        print(f"Processing {dataset_type} data: {input_file}")

        with open(input_file, 'r', encoding='utf-8') as f:
            mentions = json.load(f)

        processed_mentions = []
        batch_paths = []
        batch_indices = []

        for i, mention in enumerate(mentions):
            if 'imgPath' in mention and mention['imgPath'] != '':
                try:
                    img_name = mention['imgPath'].split('/')[-1].split('.')[0] + '.jpg'
                    img_path = os.path.join(self.args.data.mention_img_folder, img_name)
                    if os.path.exists(img_path):
                        batch_paths.append(img_path)
                        batch_indices.append(i)
                    else:
                        print(f"Image not found: {img_path}")
                        batch_indices.append(None)
                except Exception as e:
                    print(f"Error preparing image path: {e}")
                    batch_indices.append(None)
            else:
                batch_indices.append(None)

        print(f"Batch processing {len(batch_paths)} images...")
        batch_results = []
        for i in tqdm(range(0, len(batch_paths), self.batch_size), desc="Batch detection"):
            batch_end = min(i + self.batch_size, len(batch_paths))
            current_batch_paths = batch_paths[i:batch_end]

            results = self.detect_objects_batch(current_batch_paths)
            batch_results.extend(results)

        for i, mention in enumerate(mentions):
            processed_mention = mention.copy()

            if batch_indices[i] is not None and batch_indices[i] < len(batch_results):
                result = batch_results[batch_indices[i]]
                processed_mention['boxes'] = result.get('boxes', [])
            else:
                processed_mention['boxes'] = []

            processed_mentions.append(processed_mention)

        output_file = input_file[0:input_file.rfind('.')] + '_with_boxes.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_mentions, f, ensure_ascii=False, indent=2)

        print(f"{dataset_type} data with boxes saved to: {output_file}")
        return processed_mentions

    def process_all_datasets(self):
        print("Starting batch object detection for mention datasets...")

        if hasattr(self.args.data, 'train_file') and os.path.exists(self.args.data.train_file):
            self.process_mention_dataset(self.args.data.train_file, 'train')

        if hasattr(self.args.data, 'dev_file') and os.path.exists(self.args.data.dev_file):
            self.process_mention_dataset(self.args.data.dev_file, 'dev')

        if hasattr(self.args.data, 'test_file') and os.path.exists(self.args.data.test_file):
            self.process_mention_dataset(self.args.data.test_file, 'test')

        print("Batch object detection completed for all datasets!")


def main():
    from codes.utils.functions import setup_parser

    os.environ["http_proxy"] = "http://127.0.0.1:7890"
    os.environ["https_proxy"] = "http://127.0.0.1:7890"

    args = setup_parser()

    detector = BatchMentionObjectDetector(args)

    detector.process_all_datasets()


if __name__ == '__main__':
    main()