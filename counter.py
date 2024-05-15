import cv2
import numpy as np
import streamlit as st
from collections import defaultdict
from ultralytics import YOLO
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import time 



present_time = str(time.ctime(time.time()))
# Function to generate video frames
def get_video_frames_generator(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()

class Detections:
    def __init__(self, xyxy, confidence, class_id):
        self.xyxy = xyxy
        self.confidence = confidence
        self.class_id = class_id

def draw_shapes(frame):
    # Function to draw shapes on the frame
    shapes = []
    clone = frame.copy()
    num_points_selected = 0
    points = []

    def draw(event, x, y, flags, param):
        nonlocal num_points_selected, points

        if event == cv2.EVENT_LBUTTONDOWN:
            if num_points_selected == 0:
                points = [(x, y)]
                num_points_selected += 1
            elif num_points_selected < 4:
                cv2.line(clone, points[-1], (x, y), (0, 0, 255), 2)
                points.append((x, y))
                num_points_selected += 1
            elif num_points_selected == 4:
                shapes.append(points)
                cv2.polylines(clone, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)
                num_points_selected = 0
                points = []

    cv2.namedWindow("Draw Shapes (Press 'q' to quit)")
    cv2.setMouseCallback("Draw Shapes (Press 'q' to quit)", draw)

    while True:
        cv2.imshow("Draw Shapes (Press 'q' to quit)", clone)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    return shapes

st.title("Vehicle and Person Detection Web App")

col1 , col2 = st.columns(2)

uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])

graph_container = st.empty()

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    # st.sidebar.markdown("## Select Shapes")
    with col1 :
        video_player = st.video(video_path)
    with col2 :
        updated_frame_container = st.empty()

    # Load the first frame to select shapes
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if ret:
        # first_frame_container.image(frame, channels="BGR", caption="Draw shapes on the first frame")
        shapes = draw_shapes(frame)

        if shapes:
            # Initialize YOLO model
            model = YOLO("yolov8x.pt")
            model.fuse()

            # Get class names dictionary
            CLASS_NAMES_DICT = model.model.names

            # Class IDs of interest
            VEHICLE_CLASS_ID = [2, 3, 5, 7]
            PERSON_CLASS_ID = [0]

            # Create frame generator
            generator = get_video_frames_generator(video_path)

            # Initialize Plotly figure
            fig = make_subplots(rows=len(shapes), cols=1, subplot_titles=[f"Polygon {i+1}" for i in range(len(shapes))])

            # Process video frames
            vehicle_counts = defaultdict(list)
            for frame in generator:
                # Model prediction on single frame and conversion to supervision Detections
                results = model(frame)
                detections = results[0].boxes
                detections = Detections(
                    xyxy=detections.xyxy.cpu().numpy(),
                    confidence=detections.conf.cpu().numpy(),
                    class_id=detections.cls.cpu().numpy().astype(int)
                )

                # Count objects within selected shapes
                for idx, shape in enumerate(shapes, start=1):
                    vehicle_count = 0
                    for xyxy, class_id in zip(detections.xyxy, detections.class_id):
                        polygon = np.array(shape)
                        if cv2.pointPolygonTest(polygon, (xyxy[0], xyxy[1]), False) >= 0:
                            if class_id in VEHICLE_CLASS_ID:
                                vehicle_count += 1
                    vehicle_counts[idx].append(vehicle_count)

                # Annotate frame
                for shape in shapes:
                    cv2.polylines(frame, [np.array(shape)], isClosed=True, color=(0, 255, 0), thickness=2)

                # Display the frame in Streamlit
                updated_frame_container.image(frame, channels="BGR", use_column_width=True)

                # Update Plotly figure
                for idx, shape_count in vehicle_counts.items():
                    fig.add_trace(go.Scatter(y=shape_count, mode='lines', name=f'Polygon {idx}'), row=idx, col=1)

                # Update plot
                graph_container.write(fig)

