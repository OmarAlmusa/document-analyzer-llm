import os
import re
import tqdm
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import supervision as sv
import gc
import torch
import cv2


def process_image(image):
    gray_image = image.convert("L")  # Convert to grayscale
    contrast = ImageEnhance.Contrast(gray_image)
    high_contrast = contrast.enhance(2.0)  # Tune this value (2.0 = strong)
    return high_contrast

def save_pdf_images(pdf_path, output_dir, dpi=300, fmt='jpeg', quality=80, optimize=False, thread_count=1):
    os.makedirs(output_dir, exist_ok=True)
    images = convert_from_path(pdf_path, dpi=dpi, thread_count=thread_count)
    for i, img in enumerate(tqdm.tqdm(images)):
        img_path = os.path.join(output_dir, f"page_{i+1}.{fmt}")
        img.save(img_path, fmt.upper(), quality=quality, optimize=optimize)


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]


def extract_text_ocr(images_path, sort_key, reader):
    pdf_content = []
    for i, image_file in enumerate(tqdm.tqdm(sorted(os.listdir(images_path), key=sort_key))):
        image_path = os.path.join(images_path, image_file)
        result = reader.readtext(image_path)
        page_text = ''
        for _, text, _ in result:
            page_text += text

        page_content = {"text": page_text, "page_number": i+1}
        pdf_content.append(page_content)

        # Optional: free up memory every N pages
        if i % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            
    return pdf_content


def visualize_ocr_result(result, image):
    image = image.copy()
    xyxy, confidences, class_ids, label = [], [], [], []

    for idx, detection in enumerate(result):
        bbox, text, confidence = detection[0], detection[1], detection[2]
    
        # Convert bounding box format
        x_min = int(min([point[0] for point in bbox]))
        y_min = int(min([point[1] for point in bbox]))
        x_max = int(max([point[0] for point in bbox]))
        y_max = int(max([point[1] for point in bbox]))

        # Append data to lists
        xyxy.append([x_min, y_min, x_max, y_max])
        # label.append(text)
        label.append(str(idx))
        confidences.append(confidence)
        class_ids.append(0)

    detections = sv.Detections(
        xyxy=np.array(xyxy),
        confidence=np.array(confidences),
        class_id=np.array(class_ids)
    )

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(
        text_scale=2.5,      # bump this up as needed
        text_thickness=2,    # thicker font
        text_padding=5       # optional: more padding around label
    )

    annotated_image = box_annotator.annotate(scene=image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=label)

    sv.plot_image(image=annotated_image, size=(17, 22))


def draw_center_dots(result, image):
    image_copy = image.copy()
    for idx, (bbox, text, confidence) in enumerate(result):

        coordinates = (
                        int(bbox[1][0] - ((bbox[1][0]-bbox[3][0]) / 2)),
                        int(bbox[3][1] - ((bbox[3][1]-bbox[1][1]) / 2)), 
                    )

        image_copy = cv2.circle(image_copy, coordinates, radius=30, color=(0, 0, 255), thickness=-1)

        image_copy = cv2.putText(
                        image_copy, str(idx), (coordinates[0]+50, coordinates[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, fontScale=3, color=(0, 0, 255), thickness=5,
                    )
        
    return image_copy


def cal_dist(result, weights=(1.0, 1.0)):
    '''
        result: the output result of EasyOCR,
        weights: the weight of each axis alignment, (x, y)

        reminder of how images are plotted, based on that you define your weights:
                                  ^
                                  |
                                  |
                                 {-y}
                                  |
                                  |
        <-------- {-x} -------- (0,0) -------- {+x} -------->
                                  |
                                  |
                                 {+y}
                                  |
                                  |
                                  v
    '''
    # calculate the midpoint/centers of each bounding box:

    centers = [np.mean(bbox, axis=0) for bbox, _, _ in result]

    # stack midpoints as a numpy array
    coords = np.stack(centers)

    # calculate the differences between each midpoint
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]

    # calculate avg distances in axis:
    avg_x_dist = np.mean(np.abs(diff[:, :, 0]))
    avg_y_dist = np.mean(np.abs(diff[:, :, 1]))

    # calculate unit vectors
    unit_vectors = diff / (np.linalg.norm(diff, axis=-1, keepdims=True) + 1e-8)

    # normalized distances
    dist_matrix = np.linalg.norm(diff, axis=-1)

    # calculate the axis alignment scores
    x_alignment = unit_vectors[:, :, 0]
    y_alignment = unit_vectors[:, :, 1]

    x_scores = dist_matrix + weights[0] * x_alignment
    y_scores = dist_matrix + weights[1] * y_alignment

    return x_scores, y_scores, dist_matrix, (avg_x_dist, avg_y_dist)


def get_top_score(
        index, 
        scores, 
        # k=5
    ):
    sorted_indices = np.argsort(scores[index])
    sorted_indices_no_self = sorted_indices[sorted_indices != index].tolist()
    # top_predictions = []
    
    # for n, idx in enumerate(sorted_indices_no_self):
    #     top_predictions.append(idx)
    #     if n == k:
    #         break

    # return top_predictions
    return sorted_indices_no_self[0]


def directional_distance_sorting(
            result, 
            weights=(1.0, 1.0), 
            # k=5
        ):
    sorted_result = []

    first_element = max(sorted(result, key=lambda x: x[0][1][1])[:10], key=lambda x: x[0][1][0])
    sorted_result.append(first_element)

    index = result.index(first_element)

    x_scores, y_scores, dist_mat, avg_dists = cal_dist(result, weights)

    next_index = None

    for _ in range(len(result)-1):
        top_score_x = get_top_score(index, x_scores)
        fpx_dist = dist_mat[index][top_score_x]

        top_score_y = get_top_score(index, y_scores)
        fpy_dist = dist_mat[index][top_score_y]

        if fpx_dist < avg_dists[0]:
            next_index = top_score_x

        else:
            if fpy_dist < avg_dists[1]:
                next_index = top_score_y
            else:
                next_index = top_score_x

        index = next_index

        sorted_result.append(result[index])

    return sorted_result