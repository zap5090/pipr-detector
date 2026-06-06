import os
import math
import numpy as np
from ultralytics import YOLO
import cv2


model = YOLO('yolov8n.pt')
PIPE_CLASS_NAMES = {'pipe', 'tuberia', 'tubing'}
PIPE_CLASS_IDS = [idx for idx, name in model.names.items() if name.lower() in PIPE_CLASS_NAMES]
MODEL_SUPPORTS_PIPE_DETECTION = bool(PIPE_CLASS_IDS)
if not MODEL_SUPPORTS_PIPE_DETECTION:
    print('Aviso: el modelo YOLO no contiene clases de tubería. Se usará detección por formas.')


def _preprocess_gray(image, blur=(9, 9)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    return cv2.GaussianBlur(equalized, blur, 2)


def _box_iou(box_a, box_b):
    x1_a, y1_a, x2_a, y2_a = box_a
    x1_b, y1_b, x2_b, y2_b = box_b
    x1 = max(x1_a, x1_b)
    y1 = max(y1_a, y1_b)
    x2 = min(x2_a, x2_b)
    y2 = min(y2_a, y2_b)

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    area_a = max(0, x2_a - x1_a) * max(0, y2_a - y1_a)
    area_b = max(0, x2_b - x1_b) * max(0, y2_b - y1_b)
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def _get_box_from_circle(circle):
    x, y, r = circle
    return (x - r, y - r, x + r, y + r)


def _merge_circles(circles, dist_tol=40, radius_tol=15):
    if not circles:
        return []

    # Normalize circles to tuples and merge nearby duplicates
    circles = [tuple(circle) for circle in circles]
    parent = list(range(len(circles)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri = find(i)
        rj = find(j)
        if ri != rj:
            parent[rj] = ri

    for i, (x1, y1, r1) in enumerate(circles):
        for j in range(i + 1, len(circles)):
            x2, y2, r2 = circles[j]
            if abs(r1 - r2) <= radius_tol and math.hypot(x1 - x2, y1 - y2) <= dist_tol:
                union(i, j)

    clusters = {}
    for idx, circle in enumerate(circles):
        root = find(idx)
        clusters.setdefault(root, []).append(circle)

    merged = []
    for cluster in clusters.values():
        x = sum(c[0] for c in cluster) / len(cluster)
        y = sum(c[1] for c in cluster) / len(cluster)
        r = sum(c[2] for c in cluster) / len(cluster)
        merged.append((int(round(x)), int(round(y)), int(round(r))))

    return merged


def _filter_yolo_pipe_boxes(boxes, image_shape, min_area=2500, min_ratio=0.12, max_ratio=10.0):
    filtered = []
    height, width = image_shape[:2]
    for box in boxes:
        xyxy = np.asarray(box.xyxy[0]).astype(int).flatten()
        x1, y1, x2, y2 = xyxy
        if x2 <= x1 or y2 <= y1:
            continue

        w = x2 - x1
        h = y2 - y1
        area = w * h
        ratio = float(max(w, h)) / (min(w, h) + 1)
        if area < min_area or ratio < min_ratio or ratio > max_ratio:
            continue
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            continue
        filtered.append(box)
    return filtered


def detectar_tuberias_por_contornos(image, min_length=120, min_ratio=4.0, min_area=500):
    gray = _preprocess_gray(image)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pipe_boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        x, y, w, h = cv2.boundingRect(contour)
        length = max(w, h)
        ratio = float(length) / (min(w, h) + 1)
        rectangularity = area / float(w * h + 1)

        elongated_pipe = (
            length >= min_length and ratio >= min_ratio and area >= min_area
            and circularity <= 0.55 and rectangularity >= 0.25
        )
        circular_pipe = (
            area >= min_area and circularity >= 0.50 and ratio <= 1.5
        )

        if elongated_pipe or circular_pipe:
            pipe_boxes.append((x, y, w, h))

    return pipe_boxes, closed


def detectar_tuberias_por_circulos(image, dp=1.3, min_dist=70, param1=150, param2=45, min_radius=16, max_radius=180):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = _preprocess_gray(image)
    edges = cv2.Canny(blurred, 50, 130)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
                               param1=param1, param2=param2,
                               minRadius=min_radius, maxRadius=max_radius)
    if circles is None:
        return [], gray

    detected = np.round(circles[0, :]).astype(int)
    valid_circles = []
    for x, y, r in detected:
        if x - r < 0 or y - r < 0 or x + r >= image.shape[1] or y + r >= image.shape[0]:
            continue
        if r < min_radius or r > max_radius:
            continue

        mask = np.zeros_like(gray)
        cv2.circle(mask, (x, y), r, 255, 2)
        edge_pixels = cv2.countNonZero(cv2.bitwise_and(edges, edges, mask=mask))
        circumference = max(1, int(2 * np.pi * r))
        edge_ratio = edge_pixels / circumference
        if edge_pixels >= 20 and edge_ratio >= 0.22:
            valid_circles.append((x, y, r))

    valid_circles = _merge_circles(valid_circles, dist_tol=40, radius_tol=15)
    return valid_circles, gray


def _detect_yolo_pipes(image, confidence_threshold=0.35):
    if not PIPE_CLASS_IDS:
        return 0, None, None

    results = model(image)
    selected_boxes = []
    for box in results[0].boxes:
        try:
            class_id = int(box.cls)
            confidence = float(box.conf)
        except Exception:
            continue

        if class_id in PIPE_CLASS_IDS and confidence >= confidence_threshold:
            selected_boxes.append(box)

    selected_boxes = _filter_yolo_pipe_boxes(selected_boxes, image.shape)
    if not selected_boxes:
        return 0, None, None

    annotated = image.copy()
    for box in selected_boxes:
        xyxy = np.asarray(box.xyxy[0]).astype(int).flatten()
        x1, y1, x2, y2 = xyxy
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return len(selected_boxes), annotated, selected_boxes


def _detectar_tuberias(image):
    yolo_count, yolo_annotated, yolo_boxes = _detect_yolo_pipes(image)
    circle_pipes, _ = detectar_tuberias_por_circulos(image)
    contour_pipes, _ = detectar_tuberias_por_contornos(image)

    confirmed_boxes = []
    if yolo_boxes:
        for box in yolo_boxes:
            xyxy = np.asarray(box.xyxy[0]).astype(int).flatten()
            x1, y1, x2, y2 = xyxy
            pipe_box = (x1, y1, x2, y2)
            matched = False

            for circle in circle_pipes:
                if _box_iou(pipe_box, _get_box_from_circle(circle)) >= 0.25:
                    matched = True
                    break

            if not matched:
                for x, y, w, h in contour_pipes:
                    if _box_iou(pipe_box, (x, y, x + w, y + h)) >= 0.25:
                        matched = True
                        break

            if matched:
                confirmed_boxes.append(pipe_box)

        if confirmed_boxes:
            annotated_image = image.copy()
            for x1, y1, x2, y2 in confirmed_boxes:
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            return len(confirmed_boxes), annotated_image

    if circle_pipes:
        annotated_image = image.copy()
        for x, y, r in circle_pipes:
            cv2.circle(annotated_image, (x, y), r, (0, 255, 0), 2)
        return len(circle_pipes), annotated_image

    if contour_pipes:
        annotated_image = image.copy()
        for x, y, w, h in contour_pipes:
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return len(contour_pipes), annotated_image

    return 0, image.copy()


def contar_tuberias(output_path="static/detection.png"):
    cap = cv2.VideoCapture(0)
    contador = 0

    if not cap.isOpened():
        print("Error: No se puede abrir la cámara")
        return 0

    print("Enfoca el estante. Presiona 'q' para confirmar la cuenta.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        contador, annotated = _detectar_tuberias(frame)
        if annotated is None:
            annotated = frame.copy()

        cv2.putText(annotated, f"Tuberias detectadas: {contador}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("IA Contando Tuberias", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(output_path, annotated)
            break

    cap.release()
    cv2.destroyAllWindows()
    return contador


def procesar_imagen(input_path, output_path="static/detection.png"):
    image = cv2.imread(input_path)
    if image is None:
        print("Error: no se puede leer la imagen de entrada")
        return 0

    contador, annotated = _detectar_tuberias(image)
    if annotated is None:
        annotated = image.copy()

    cv2.putText(annotated, f"Tuberias detectadas: {contador}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(output_path, annotated)
    return contador