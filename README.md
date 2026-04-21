# Object Detection and Tracking with YOLOv8 and ByteTrack

> A project for detecting, estimating speed, and counting vehicles in videos using YOLOv8 and ByteTrack.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Why This Exists

This project addresses the need for accurate and efficient vehicle detection and tracking in real-time video feeds. It leverages the power of YOLOv8 for object detection and ByteTrack for effective tracking, making it suitable for various applications in traffic monitoring and automated surveillance.

## Quick Start

Get started with just a few commands:

1. Clone the repository:
   ```bash
   git clone https://github.com/Huyy1510/Research-and-Using-Object-Detection-and-Object-Tracking.git
   cd Research-and-Using-Object-Detection-and-Object-Tracking
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py --input <video_file> --output <output_file>
   ```

## Installation

**Prerequisites**: Ensure you have Python 3.8+ and pip installed.

1. Clone the repository:
   ```bash
   git clone https://github.com/Huyy1510/Research-and-Using-Object-Detection-and-Object-Tracking.git
   cd Research-and-Using-Object-Detection-and-Object-Tracking
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Example

To run the object detection and tracking on a video file:

```bash
python app.py --input path/to/video.mp4 --output path/to/output.mp4
```

### Configuration Options

| Option     | Type    | Default          | Description                                      |
|------------|---------|------------------|--------------------------------------------------|
| `--input`  | string  | `None`           | Path to the input video file.                    |
| `--output` | string  | `None`           | Path to save the output video with detections.   |
| `--model`  | string  | `yolov8.pt`      | Path to the YOLOv8 model weights.                |

## API Reference

For more details on the available functions and their parameters, refer to the source code in the repository.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to this project.

## License

MIT © [Huyy1510](https://github.com/Huyy1510)