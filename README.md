# Research and Using Object Detection and Object Tracking

This repository demonstrates the application of YOLOv8 and ByteTrack for detecting, estimating the speed, and counting vehicles in video footage.

## Why This Project Exists

In the realm of intelligent transportation systems, accurately detecting and tracking vehicles is crucial for traffic management, urban planning, and safety. This project leverages state-of-the-art object detection and tracking algorithms to provide real-time insights into vehicular movement.

## Quick Start

To get started quickly, follow these steps:

### Prerequisites

- Python 3.8+
- OpenCV
- TensorFlow
- YOLOv8 model files

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Huyy1510/Research-and-Using-Object-Detection-and-Object-Tracking.git
   cd Research-and-Using-Object-Detection-and-Object-Tracking
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the YOLOv8 model weights. You can find the weights [here](https://github.com/ultralytics/yolov5/releases).

### Running the Application

To run the object detection and tracking on a video file, use the following command:
```bash
python main.py --source path_to_your_video.mp4
```

Replace `path_to_your_video.mp4` with the path to your video file.

## Usage

- The application will process the video and display the detected vehicles with bounding boxes.
- Speed estimation and vehicle counting will be displayed on the video feed.

## API Reference

- `main.py`: Entry point for running the object detection and tracking.
- `yolo_model.py`: Contains functions for loading and running the YOLO model.
- `tracker.py`: Implements tracking logic using ByteTrack.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for more details on how to contribute to this project.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any inquiries, please reach out to [Huyy1510](https://github.com/Huyy1510).