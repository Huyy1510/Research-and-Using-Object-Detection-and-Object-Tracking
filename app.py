import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from process import process_video, process_video_frame, process_video_frame_2, process_video_2
import os
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import supervision as sv
from streamlit_image_coordinates import streamlit_image_coordinates
st.set_page_config(page_title="Vehicle Speed Dashboard", layout="centered")
st.title("Vehicle Speed & Count Dashboard")



# Thư mục dữ liệu
upload_dir = Path("data/uploads")
output_dir = Path("data/outputs")
log_dir = Path("data/logs")

upload_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)
log_dir.mkdir(parents=True, exist_ok=True)

#uploaded_file = st.file_uploader("Upload video (.mp4 hoặc .avi)", type=["mp4", "avi"])

current_time = datetime.now()
# Sử dụng session state để lưu trạng thái
if "video_processed" not in st.session_state:
    st.session_state["video_processed"] = False
if "output_path" not in st.session_state:
    st.session_state["output_path"] = ""
if "log_path" not in st.session_state:
    st.session_state["log_path"] = ""
if "webcam_running" not in st.session_state:
    st.session_state["webcam_running"] = False
if "frames_raw" not in st.session_state:
    st.session_state["frames_raw"] = []
if "frames_annotated" not in st.session_state:
    st.session_state["frames_annotated"] = []

video_source = st.radio("Nguồn video:", ["Upload file", "Webcam trực tiếp"])
if video_source == "Upload file":
    uploaded_file = st.file_uploader("Upload video (.mp4 hoặc .avi)", type=["mp4", "avi"])
    if uploaded_file:
        input_path = upload_dir / f'{current_time.strftime("%Y%m%d_%H%M%S")}_{uploaded_file.name}'
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Đã upload: {uploaded_file.name}")

        # Lấy frame đầu tiên để chọn ROI
        cap = cv2.VideoCapture(str(input_path))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            st.error("Không đọc được frame đầu!")
            st.stop()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # ==== Resize frame để chọn ROI, luôn vừa màn hình, giữ tỷ lệ ====
        max_width = 1000  # Có thể chỉnh thành 1000 tùy ý bạn
        orig_h, orig_w = frame_rgb.shape[:2]
        if orig_w > max_width:
            scale = max_width / orig_w
            new_w = max_width
            new_h = int(orig_h * scale)
            resized_img = Image.fromarray(frame_rgb).resize((new_w, new_h))
        else:
            scale = 1.0
            resized_img = Image.fromarray(frame_rgb)
        st.write("Hãy click chọn 4 điểm theo thứ tự trên ảnh để xác định vùng ROI.")
        coords = streamlit_image_coordinates(resized_img, key="roi_img")

        # Reset lại điểm khi upload video mới
        if "last_uploaded" not in st.session_state or st.session_state["last_uploaded"] != uploaded_file.name:
            st.session_state["roi_points"] = []
            st.session_state["last_uploaded"] = uploaded_file.name

        if coords is not None and len(st.session_state["roi_points"]) < 4:
            # Convert về tọa độ gốc
            x = int(coords["x"] / scale)
            y = int(coords["y"] / scale)
            st.session_state["roi_points"].append([x, y])

        for i, pt in enumerate(st.session_state["roi_points"]):
            st.write(f"Điểm {i+1}: {pt}")

        if len(st.session_state["roi_points"]) < 4:
            st.info("Chưa đủ 4 điểm, hãy tiếp tục click trên ảnh.")
            st.stop()
        else:
            st.success("Đã chọn đủ 4 điểm vùng ROI!")
            roi_points = np.array(st.session_state["roi_points"])

            output_path = output_dir / f"annotated_{uploaded_file.name}_{current_time.strftime('%Y%m%d_%H%M%S')}.mp4"
            log_path = log_dir / f"log_{uploaded_file.name.replace('.mp4', '.csv')}"
        
        #-----------------TEST----------------
        # video_info = sv.VideoInfo.from_video_path(str(input_path))
        # roi_points = [
        #     [0, 0],
        #     [video_info.width - 1, 0],
        #     [video_info.width - 1, video_info.height - 1],
        #     [0, video_info.height - 1]
        # ]
        #-----------------END TEST----------------

            # Tạo image placeholder và callback
            image_placeholder = st.empty()
            def show_frame_on_web(annotated_frame):
                image_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")

            # Gọi hàm process_video với callback
            if st.button("Xử lý và xem realtime"):
                with st.spinner("Đang xử lý video và phát realtime..."):
                    process_video_2(
                        str(input_path),
                        str(output_path),
                        str(log_path),
                        roi_points=roi_points,
                        model_path="yolov8ft.pt",
                        frame_callback=show_frame_on_web
                    )
                st.success("Hoàn tất xử lý video!")
            
        #-----------Tắt tạm-------------
        # # Chỉ xử lý nếu chưa xử lý hoặc file chưa tồn tại
        #     if not st.session_state["video_processed"] or not output_path.exists() or not log_path.exists():
        #         with st.spinner("Đang xử lý video..."):
        #             process_video(str(input_path), str(output_path), str(log_path), roi_points=roi_points, model_path="yolov8ft.pt")
        #         st.session_state["video_processed"] = True
        #         st.session_state["output_path"] = str(output_path)
        #         st.session_state["log_path"] = str(log_path)
        #         st.success("Hoàn tất xử lý video!")
        #     else:
        #         st.success("Video đã được xử lý trước đó.")

        #     # Hiển thị video kết quả (chỉ khi file > 0 bytes)
        #     # Hiển thị video kết quả
        #     if output_path.exists() and os.path.getsize(output_path) > 0:
        #         st.subheader("Kết quả video")
        #         with open(output_path, "rb") as video_file:
        #             video_bytes = video_file.read()
        #         st.video(video_bytes)
        #------------------------
            if output_path.exists() and os.path.getsize(output_path) > 0:
                st.subheader("Kết quả video")
                with open(output_path, "rb") as video_file:
                    video_bytes = video_file.read()
                st.video(video_bytes)
            # Load log và biểu đồ
            if log_path.exists():
                df = pd.read_csv(log_path)

                # Số lượng xe theo thời gian (giây)
                count_by_sec = df.groupby("timestamp")["object_id"].nunique()

                st.subheader("Số lượng xe theo thời gian")
                fig, ax = plt.subplots()
                count_by_sec.plot(ax=ax)
                ax.set_xlabel("Thời gian (giây)")
                ax.set_ylabel("Số lượng xe")
                st.pyplot(fig)

                # Nút tải file log (không gây reload lại)
                st.download_button(
                    "Tải file log CSV",
                    df.to_csv(index=False),
                    file_name=log_path.name,
                    mime="text/csv"
                )


elif video_source == "Webcam trực tiếp":
    st.info("Nhấn 'Bắt đầu' để realtime. Nhấn 'Dừng' để dừng và xuất file.")
    col1, col2 = st.columns(2)
    start = col1.button("Bắt đầu")
    stop = col2.button("Dừng")
    image_placeholder = st.empty()

    if start and not st.session_state["webcam_running"]:
        st.session_state["webcam_running"] = True
        st.session_state["frames_raw"] = []
        st.session_state["frames_annotated"] = []

    if st.session_state["webcam_running"]:
        cap = cv2.VideoCapture(0)
        fps = 20
        ret, frame = cap.read()
        if not ret:
            st.error("Không lấy được frame webcam.")
            cap.release()
            st.session_state["webcam_running"] = False
        else:
            h, w = frame.shape[:2]
            roi_points = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
            while st.session_state["webcam_running"]:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Không lấy được frame từ webcam.")
                    break
                annotated_frame, _ = process_video_frame(frame, roi_points)
                st.session_state["frames_raw"].append(frame.copy())
                st.session_state["frames_annotated"].append(annotated_frame.copy())
                image_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")
                if stop:
                    st.session_state["webcam_running"] = False
                    break
            cap.release()
            # Lưu file khi dừng
            if len(st.session_state["frames_annotated"]) > 0:
                now = datetime.now().strftime("%Y%m%d_%H%M%S")
                h, w = st.session_state["frames_annotated"][0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fourcc_ann = cv2.VideoWriter_fourcc(*'avc1')
                video_raw_path = upload_dir / f"webcam_raw_{now}.mp4"
                video_annotated_path = output_dir / f"webcam_annotated_{now}.mp4"
                writer_raw = cv2.VideoWriter(str(video_raw_path), fourcc, fps, (w, h))
                writer_ann = cv2.VideoWriter(str(video_annotated_path), fourcc_ann, fps, (w, h))
                for f_raw, f_ann in zip(st.session_state["frames_raw"], st.session_state["frames_annotated"]):
                    writer_raw.write(f_raw)
                    writer_ann.write(f_ann)
                writer_raw.release()
                writer_ann.release()
                st.success("Đã lưu video webcam gốc và annotated!")

                # Hiển thị lại video annotated
                with open(video_annotated_path, "rb") as vfile:
                    st.video(vfile.read())

                # Reset lại list frame
                st.session_state["frames_raw"] = []
                st.session_state["frames_annotated"] = []       