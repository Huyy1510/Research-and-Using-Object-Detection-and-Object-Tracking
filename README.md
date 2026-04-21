# Research and Using Object Detection and Object Tracking

> This project leverages YOLOv8 and ByteTrack for vehicle detection, speed estimation, and counting in video footage.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Why This Exists

This project addresses the need for efficient and accurate vehicle detection and tracking in real-time video streams. By utilizing state-of-the-art algorithms, it simplifies the process of monitoring traffic and analyzing vehicle behavior.

## Quick Start

To get started quickly, follow these steps to install the necessary dependencies and run the application.

### Installation

**Prerequisites**: Python 3.8+, pip

1. **Install the Ultralytics package**:

```bash
pip install ultralytics
```

2. **Clone the repository**:

```bash
git clone https://github.com/Huyy1510/Research-and-Using-Object-Detection-and-Object-Tracking.git
cd Research-and-Using-Object-Detection-and-Object-Tracking
```

3. **Install additional dependencies** (if required):

```bash
pip install -r requirements.txt
```

## Usage

### Basic Example

To run the object detection and tracking on a video file, execute the following command:

```bash
python app.py --source path/to/video.mp4
```

### Configuration

You can configure the settings in `bytetrack.yaml`. Key options include:

| Option          | Type    | Default      | Description                                |
|-----------------|---------|--------------|--------------------------------------------|
| `video_source`  | string  | `0`          | Source of the video (0 for webcam)        |
| `output_format` | string  | `output.mp4` | Name of the output video file              |
| `confidence`    | float   | `0.25`       | Confidence threshold for detection         |

### Advanced Usage

For advanced configuration, modify the parameters in `bytetrack.yaml` to tailor the detection and tracking behavior to your needs.

## API Reference

See [full API reference →](https://github.com/Huyy1510/Research-and-Using-Object-Detection-and-Object-Tracking)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## License

MIT © [Huyy1510](https://github.com/Huyy1510)