# Real-time Object Detection using YOLOv5

## Project Description
This project implements a comprehensive real-time object detection system using YOLOv5 (You Only Look Once version 5). The system provides live object detection with detailed bounding boxes, class labels, confidence scores, and real-time statistics. Designed for internship demonstration purposes, it showcases advanced computer vision capabilities with professional visualization features.

## 1. Project Objective
Develop a robust real-time object detection system that can:

- Detect multiple objects simultaneously with high accuracy
- Provide real-time performance statistics and FPS monitoring
- Display comprehensive detection summaries and class distributions
- Offer intuitive visualization with color-coded bounding boxes
- Include user-friendly controls for interaction and frame capture

## 2. Technical Specifications
- **Model**: YOLOv5su (Small Ultra version)
- **Framework**: Ultralytics YOLO with OpenCV integration
- **Processing Speed**: Real-time (30 FPS target)
- **Input Sources**: Webcam (default) or video files
- **Output**: Annotated video stream with detection overlays

## 3. Methodology

### Model Architecture
- **Backbone**: CSPDarknet53 for feature extraction
- **Neck**: PANet for feature pyramid network
- **Head**: YOLO detection head with anchor boxes
- **Input Resolution**: 640x480 pixels (optimized for performance)

### Detection Pipeline
1. **Frame Capture**: Real-time video stream processing
2. **Preprocessing**: Automatic resizing and normalization
3. **Inference**: YOLOv5 model prediction
4. **Post-processing**: Non-Maximum Suppression (NMS) and confidence filtering
5. **Visualization**: Bounding boxes, labels, and statistics overlay

### Advanced Features
- **Color-coded Detection**: Different colors for various object classes
- **Confidence Visualization**: Dynamic bounding box thickness and confidence bars
- **Real-time Statistics**: FPS monitoring and object count tracking
- **Detection Summary**: Class-wise aggregation and average confidence
- **Interactive Controls**: Pause, resume, and frame capture functionality

## 4. System Features

### Core Detection Capabilities
- **Multi-class Detection**: 80+ COCO dataset classes including:
  - Persons, vehicles, electronic devices, furniture, etc.
- **Real-time Processing**: Optimized for live video streams
- **Confidence Thresholding**: Minimum 25% confidence for reliable detections

### Visualization Features
- **Bounding Boxes**: Color-coded by object class with confidence-based thickness
- **Information Panels**:
  - Left-side statistics panel with FPS and object counts
  - Right-side detection summary with class aggregations
- **Confidence Bars**: Visual indicators of detection certainty
- **Semi-transparent Overlays**: Professional UI design

### User Controls
- **'q'**: Quit application
- **'s'**: Save current frame as image
- **'p'**: Pause/resume detection

## 5. Performance Metrics
- **Frame Rate**: 30 FPS target (hardware dependent)
- **Detection Accuracy**: Based on YOLOv5su pre-trained weights
- **Confidence Range**: 0.25 to 1.0 threshold
- **Multi-object Handling**: Simultaneous detection of multiple object types

## 6. Business Applications

### Security and Surveillance
- Real-time monitoring and intrusion detection
- People counting and crowd analysis
- Suspicious activity identification

### Retail and Analytics
- Customer behavior analysis
- Product placement optimization
- Inventory management through object counting

### Industrial Applications
- Quality control and defect detection
- Safety compliance monitoring
- Process automation through visual inspection

### Educational Use
- Computer vision training and demonstrations
- AI/ML internship projects
- Research and development prototyping

## 7. Project Setup and Requirements

### Requirements
- Python 3.7+
- OpenCV (cv2)
- Ultralytics YOLO
- NumPy
- Webcam or video source

### Installation
Install dependencies by running:

```bash
pip install opencv-python ultralytics numpy
```

### Model Download
The system automatically downloads YOLOv5su weights on first run (approximately 14MB).

### Running the Project
1. Ensure webcam is connected or video file path is specified
2. Run the main script:
```bash
python v5.py
```

### System will:
- Initialize YOLOv5 model
- Start video capture from specified source
- Display real-time detection interface
- Process frames at optimal speed

## 8. Code Structure

### Main Classes
- **ObjectDetector**: Core detection and visualization class
- **Methods**:
  - `__init__()`: Model initialization and color scheme setup
  - `calculate_fps()`: Real-time performance monitoring
  - `process_frame()`: Main detection pipeline
  - `run_detection()`: Video stream management

### Key Components
- **Color Management**: Class-specific color coding
- **Statistics Panel**: Real-time performance metrics
- **Detection Summary**: Aggregate object information
- **Visualization Tools**: Professional annotation system

## 9. Customization Options

### Model Selection
- Switch between YOLOv5 variants (n, s, m, l, x) by changing model path
- Custom trained models supported

### Detection Parameters
- Adjust confidence threshold (currently 0.25)
- Modify bounding box appearance
- Customize color schemes for different classes

### Output Options
- Save detection results as video
- Export frame-by-frame analytics
- Integrate with other applications via API

## 10. Future Enhancements

### Technical Improvements
- Implement object tracking across frames
- Add distance estimation capabilities
- Integrate with deep sort for improved tracking
- Add support for custom datasets

### Feature Additions
- Web interface for remote monitoring
- Mobile app integration
- Cloud-based processing for multiple streams
- Alert system for specific object detection

### Performance Optimizations
- GPU acceleration support
- Multi-threaded processing
- Model quantization for edge devices
- Batch processing for improved efficiency

## 11. Troubleshooting

### Common Issues
- **Webcam not detected**: Try different source indices (0, 1, 2)
- **Low FPS**: Reduce frame resolution or use lighter model variant
- **Model download issues**: Check internet connection and firewall settings

### Performance Tips
- Use GPU if available for faster inference
- Close unnecessary applications during execution
- Ensure adequate lighting for better detection accuracy

## 12. Contact
For questions or collaboration:
- **Name**: Ghanashyam T V
- **Email**: ghanashyamtv16@gmail.com
- **LinkedIn**: [linkedin.com/in/ghanashyam-tv](https://linkedin.com/in/ghanashyam-tv)

---

Thank you for exploring the Real-time Object Detection System! This project demonstrates advanced computer vision capabilities with practical applications across various industries. The system provides a solid foundation for further development and customization based on specific use cases.

---