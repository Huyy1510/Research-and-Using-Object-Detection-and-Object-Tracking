import cv2
from collections import defaultdict
import supervision as sv
from ultralytics import YOLO
from ultralytics.solutions import SpeedEstimator
import numpy as np
from supervision.assets import download_assets, VideoAssets
from collections import deque
import pandas as pd
from PIL import Image
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel

def process_video_2(
    input_path, 
    output_path, 
    log_path, 
    roi_points,
    model_path="yolov8ft.pt", 
    frame_callback=None,
    slice_height=640, 
    slice_width=640, 
    overlap_height_ratio=0.2, 
    overlap_width_ratio=0.2,
    min_area_threshold=30*30  # object nhỏ hơn ngưỡng này → dùng SAHI
):
    source = np.array(roi_points)

    # Model YOLO thường
    yolo_model = YOLO(model_path)
    class_names = yolo_model.model.names

    # Model SAHI wrapper
    sahi_model = AutoDetectionModel.from_pretrained(
        model_path=model_path,
        model_type='yolov8',
        confidence_threshold=0.25,
        device="cuda"  # hoặc "cpu" nếu không có GPU
    )

    path = input_path
    video_info = sv.VideoInfo.from_video_path(path)
    byte_track = sv.ByteTrack(frame_rate=video_info.fps)

    thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
    bounding_box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_thickness=thickness,
        text_scale=text_scale,
        text_position=sv.Position.TOP_CENTER
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness, 
        trace_length=video_info.fps, 
        position=sv.Position.BOTTOM_CENTER
    )

    frame_generator = sv.get_video_frames_generator(path)
    polygon_zone = sv.PolygonZone(source)

    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
    frame_id = 0
    logs = []

    with sv.VideoSink(target_path=output_path, video_info=video_info, codec='avc1') as sink:
        for frame in frame_generator:
            # --- YOLO thường ---
            result = yolo_model(frame)[0]
            detections = sv.Detections.from_ultralytics(result)

            # Kiểm tra object nhỏ
            small_object = False
            for box in detections.xyxy:
                x1, y1, x2, y2 = box
                area = (x2 - x1) * (y2 - y1)
                if area < min_area_threshold:
                    small_object = True
                    break

            # --- Nếu có object nhỏ → chạy lại bằng SAHI ---
            if small_object:
                sahi_result = get_sliced_prediction(
                    frame,
                    sahi_model,
                    slice_height=slice_height,
                    slice_width=slice_width,
                    overlap_height_ratio=overlap_height_ratio,
                    overlap_width_ratio=overlap_width_ratio
                )
                boxes, scores, class_ids = [], [], []
                for obj in sahi_result.object_prediction_list:
                    x1, y1, x2, y2 = obj.bbox.to_xyxy()
                    boxes.append([x1, y1, x2, y2])
                    scores.append(obj.score.value)
                    class_ids.append(obj.category.id)
                if boxes:  # chỉ update nếu có kết quả
                    detections = sv.Detections(
                        xyxy=np.array(boxes),
                        confidence=np.array(scores),
                        class_id=np.array(class_ids)
                    )

            # --- ROI + Tracking ---
            detections = detections[polygon_zone.trigger(detections)]
            detections = byte_track.update_with_detections(detections=detections)
            object_count = len(detections)

            # --- Tính tốc độ ---
            time_sec = round(frame_id / video_info.fps, 2)
            labels = []
            meter_per_pixel = 0.05
            for tracker_id, class_id, box in zip(detections.tracker_id, detections.class_id, detections.xyxy):
                x1, y1, x2, y2 = box
                y = (y1 + y2) / 2
                coordinates[tracker_id].append(y)
                if len(coordinates[tracker_id]) < video_info.fps / 2:
                    labels.append(f"#{tracker_id} {class_names[class_id]}")
                else:
                    coor_start = coordinates[tracker_id][-1]
                    coor_end = coordinates[tracker_id][0]
                    distance_pixel = abs(coor_start - coor_end)
                    distance_meter = distance_pixel * meter_per_pixel
                    time = len(coordinates[tracker_id]) / video_info.fps
                    speed = distance_meter / time * 3.6
                    labels.append(f"#{tracker_id} {class_names[class_id]} {speed:.2f} km/h")

                    logs.append({
                        "frame_id": frame_id,
                        "timestamp": time_sec,
                        "object_id": tracker_id,
                        "speed_kmph": speed,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    })
            frame_id += 1

            # --- Annotate frame ---
            annotated_frame = frame.copy()
            annotated_frame = sv.draw_polygon(scene=annotated_frame, polygon=source, color=sv.Color.RED)
            annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = bounding_box_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

            text = f"Objects Count: {object_count}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = annotated_frame.shape[1] - text_width - 30
            text_y = 40
            cv2.putText(annotated_frame, text, (text_x, text_y), font, font_scale, (255, 0, 0), thickness, cv2.LINE_AA)

            if frame_callback is not None:
                frame_callback(annotated_frame)
            sink.write_frame(frame=annotated_frame)

    df = pd.DataFrame(logs)
    df.to_csv(log_path, index=False)

def process_video(input_path, output_path, log_path, roi_points , model_path="yolov8ft.pt", frame_callback=None):
    #source = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]]) #tim cach cho nguoi dung tuy chinh
    source = np.array(roi_points)   # Mặc định là toàn khung hình
    model = YOLO(model_path)
    class_names = model.model.names
    path = input_path
    # Đảm bảo dùng H.264 cho output mp4
    # if output_path.endswith('.mp4'):
    #     fourcc = cv2.VideoWriter_fourcc(*'avc1')  #streamlit run app.py H.264
    # else:
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    video_info = sv.VideoInfo.from_video_path(path)

    #writer = cv2.VideoWriter(output_path, fourcc, video_info.fps, (video_info.width, video_info.height))

    byte_track = sv.ByteTrack(frame_rate=video_info.fps)
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(
        resolution_wh=video_info.resolution_wh)
    bounding_box_annotator = sv.BoxAnnotator(thickness = thickness)
    label_annotator = sv.LabelAnnotator(
        text_thickness = thickness,
        text_scale = text_scale,
        text_position= sv.Position.TOP_CENTER
    )
    trace_annotator = sv.TraceAnnotator(thickness=thickness, trace_length=video_info.fps, position=sv.Position.BOTTOM_CENTER)
    frame_generator = sv.get_video_frames_generator(path)

    polygon_zone = sv.PolygonZone(source)

    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
    frame_id = 0
    logs = []

    with sv.VideoSink(target_path=output_path, video_info=video_info, codec= 'avc1') as sink:
        for frame in frame_generator:
            result = model(frame)[0]  #lấy chỉ số box
            detections = sv.Detections.from_ultralytics(result)
            detections = detections[polygon_zone.trigger(detections)]
            detections = byte_track.update_with_detections(detections=detections)
            object_count= len(detections)
            
            time_sec = round(frame_id / video_info.fps, 2)
            labels = []
            meter_per_pixel = 0.05
            for tracker_id,class_id, box in zip(detections.tracker_id,detections.class_id, detections.xyxy):
                x1, y1, x2, y2 = box
                y = (y1 + y2) / 2   # tâm theo trục y, hoặc dùng y2 (đáy bbox)
                coordinates[tracker_id].append(y)
                if len(coordinates[tracker_id]) < video_info.fps / 2:
                    labels.append(f"#{tracker_id} {class_names[class_id]}")
                else:
                    coor_start = coordinates[tracker_id][-1]
                    coor_end = coordinates[tracker_id][0]
                    distance_pixel = abs(coor_start - coor_end)
                    distance_meter = distance_pixel * meter_per_pixel
                    time = len(coordinates[tracker_id]) / video_info.fps
                    speed = distance_meter / time * 3.6  # speed in km/h
                    labels.append(f"#{tracker_id} {class_names[class_id]} {speed:.2f} km/h")

                    logs.append({
                        "frame_id": frame_id,
                        "timestamp": time_sec,
                        "object_id": tracker_id,
                        "speed_kmph": speed,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    })
            frame_id += 1
            annotated_frame = frame.copy()
            annotated_frame = sv.draw_polygon(
                scene = annotated_frame,
                polygon = source,
                color = sv.Color.RED
            )
            annotated_frame = trace_annotator.annotate(
                scene = annotated_frame,
                detections = detections
            )
            annotated_frame = bounding_box_annotator.annotate(
                scene = annotated_frame,
                detections = detections
            )
            annotated_frame = label_annotator.annotate(
                scene = annotated_frame,
                detections = detections,
                labels = labels
            )
            text = f"Objects Count: {object_count}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = annotated_frame.shape[1] - text_width - 30  # Cách 30px lề phải
            text_y = 40  # Cách 40px lề trên
            
            cv2.putText(
                annotated_frame,
                text,
                (text_x, text_y),
                font,
                font_scale,
                (255, 0, 0),  # Màu đỏ
                thickness,
                cv2.LINE_AA
            )
            
            
            if frame_callback is not None:
                frame_callback(annotated_frame)
            sink.write_frame(frame=annotated_frame)
            #writer.write(annotated_frame)
            # annotated_frame = cv2.resize(annotated_frame, (1280, 720))
            # cv2.imshow("annotated", annotated_frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
    #writer.release()
    #cv2.destroyAllWindows()

    df = pd.DataFrame(logs)
    df.to_csv(log_path, index=False)




# Khởi tạo model 1 lần (bên ngoài hàm nếu gọi liên tục)
model = YOLO("yolov8ft.pt")
class_names = model.model.names

def process_video_frame(frame, roi_points, meter_per_pixel=0.05):
    source = np.array(roi_points)
    result = model(frame)[0]
    detections = sv.Detections.from_ultralytics(result)
    polygon_zone = sv.PolygonZone(source)
    detections = detections[polygon_zone.trigger(detections)]
    object_count = len(detections) if detections is not None else 0

    annotated_frame = frame.copy()
    annotated_frame = sv.draw_polygon(
        scene=annotated_frame,
        polygon=source,
        color=sv.Color.RED
    )
    bounding_box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(
        text_thickness=2,
        text_scale=1,
        text_position=sv.Position.TOP_CENTER
    )
    # --- Sửa chỗ này ---
    labels = []
    try:
        # Nếu có class_id, dùng class name
        for class_id in detections.class_id:
            labels.append(class_names[class_id])
    except Exception:
        labels = ["object"] * len(detections)
    if len(labels) != len(detections):
        labels = ["object"] * len(detections)
    # -------------------

    annotated_frame = bounding_box_annotator.annotate(
        scene=annotated_frame,
        detections=detections
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels
    )
    # ... phần còn lại giữ nguyên ...
    # Hiển thị số lượng object góc phải trên
    text = f"Objects Count: {object_count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = annotated_frame.shape[1] - text_width - 30
    text_y = 40
    cv2.putText(
        annotated_frame,
        text,
        (text_x, text_y),
        font,
        font_scale,
        (255, 0, 0),
        thickness,
        cv2.LINE_AA
    )
    return annotated_frame, object_count



def process_video_frame_2(frame, roi_points, frame_id=0, fps=20, meter_per_pixel=0.05):
    """
    Xử lý 1 frame: detect, annotate, tracking, log speed.
    Trả về: annotated_frame, object_count, logs (list)
    """
    source = np.array(roi_points)
    result = model(frame)[0]
    detections = sv.Detections.from_ultralytics(result)
    polygon_zone = sv.PolygonZone(source)
    detections = detections[polygon_zone.trigger(detections)]
    object_count = len(detections) if detections is not None else 0

    bounding_box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(
        text_thickness=2,
        text_scale=1,
        text_position=sv.Position.TOP_CENTER
    )
    trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=fps, position=sv.Position.BOTTOM_CENTER)
    annotated_frame = frame.copy()
    annotated_frame = sv.draw_polygon(
        scene=annotated_frame,
        polygon=source,
        color=sv.Color.RED
    )
    # annotated_frame = trace_annotator.annotate(
    #     scene=annotated_frame,
    #     detections=detections
    # )
    annotated_frame = bounding_box_annotator.annotate(
        scene=annotated_frame,
        detections=detections
    )

    # Labels và log cho từng object
    logs = []
    labels = []
    time_sec = round(frame_id / fps, 2)
    # Nếu có tracker_id/class_id thì show, nếu không thì label mặc định
    try:
        for i in range(len(detections)):
            try:
                tracker_id = detections.tracker_id[i]
                class_id = detections.class_id[i]
                box = detections.xyxy[i]
                x1, y1, x2, y2 = box
                labels.append(f"#{tracker_id} {class_names[class_id]}")
                logs.append({
                    "frame_id": frame_id,
                    "timestamp": time_sec,
                    "object_id": tracker_id,
                    "speed_kmph": "",  # Nếu cần tính speed thì bổ sung tracking
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                })
            except Exception:
                labels.append("object")
    except Exception:
        labels = ["object"] * len(detections)
    if len(labels) != len(detections):
        labels = ["object"] * len(detections)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels
    )
    # Hiển thị số lượng object góc phải trên
    text = f"Objects Count: {object_count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = annotated_frame.shape[1] - text_width - 30
    text_y = 40
    cv2.putText(
        annotated_frame,
        text,
        (text_x, text_y),
        font,
        font_scale,
        (255, 0, 0),
        thickness,
        cv2.LINE_AA
    )
    return annotated_frame, object_count, logs