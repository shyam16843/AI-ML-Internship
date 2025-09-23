import cv2
import numpy as np
from ultralytics import YOLO
import time
from typing import List, Tuple, Dict

class ObjectDetector:
    """
    Real-time Object Detection using YOLOv5
    Enhanced version for internship demonstration
    """
    
    def __init__(self, model_path: str = 'yolov5su.pt'):
        """Initialize the object detector with YOLOv5 model"""
        self.model = YOLO(model_path)
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Color scheme for different classes
        self.colors = {
            'person': (0, 255, 0),      # Green
            'car': (255, 0, 0),         # Blue
            'cell phone': (0, 0, 255),  # Red
            'laptop': (255, 0, 255),    # Magenta
            'bottle': (0, 255, 255),    # Yellow
            'chair': (255, 165, 0),     # Orange
            'default': (255, 255, 0)    # Cyan for other classes
        }
    
    def calculate_fps(self) -> float:
        """Calculate and return current FPS"""
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        
        if elapsed_time > 1.0:  # Update FPS every second
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
        
        return self.fps
    
    def get_color_for_class(self, class_name: str) -> Tuple[int, int, int]:
        """Get color for specific object class"""
        return self.colors.get(class_name, self.colors['default'])
    
    def draw_detection_info(self, frame: np.ndarray, box: np.ndarray, 
                          class_name: str, confidence: float) -> None:
        """Draw bounding box and information for detected object"""
        x1, y1, x2, y2 = map(int, box)
        color = self.get_color_for_class(class_name)
        
        # Draw bounding box with thickness based on confidence
        thickness = max(1, int(confidence * 3))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Create label with class name and confidence
        label = f'{class_name}: {confidence:.2f}'
        
        # Calculate text background size
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        
        # Draw text background
        cv2.rectangle(frame, (x1, y1 - text_height - 10), 
                     (x1 + text_width, y1), color, -1)
        
        # Draw text
        cv2.putText(frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw confidence bar
        bar_width = x2 - x1
        confidence_width = int(bar_width * confidence)
        cv2.rectangle(frame, (x1, y2 + 5), (x1 + confidence_width, y2 + 10), color, -1)
        cv2.rectangle(frame, (x1, y2 + 5), (x2, y2 + 10), (255, 255, 255), 1)
    
    def draw_statistics(self, frame: np.ndarray, detections: List[Dict]) -> None:
        """Draw statistics panel on the frame"""
        fps = self.calculate_fps()
        
        # Count objects by class
        class_counts = {}
        for detection in detections:
            class_name = detection['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Create statistics text
        stats = [
            f"FPS: {fps:.1f}",
            f"Total Objects: {len(detections)}",
            f"Model: YOLOv5su",
            f"Classes Detected: {len(class_counts)}"
        ]
        
        # Add top 3 most detected classes
        if class_counts:
            top_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            stats.append("Top Classes:")
            for class_name, count in top_classes:
                stats.append(f"  {class_name}: {count}")
        
        # Draw semi-transparent background for stats
        stats_height = len(stats) * 25 + 20
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, stats_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw statistics text
        for i, stat in enumerate(stats):
            y_position = 35 + i * 25
            font_scale = 0.5 if i > 3 else 0.6
            color = (255, 255, 255) if not stat.startswith("  ") else (200, 200, 200)
            cv2.putText(frame, stat, (15, y_position), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
    
    def draw_detection_summary(self, frame: np.ndarray, detections: List[Dict]) -> None:
        """Draw detection summary on the right side"""
        if not detections:
            return
            
        # Group detections by class
        class_detections = {}
        for detection in detections:
            class_name = detection['class']
            if class_name not in class_detections:
                class_detections[class_name] = []
            class_detections[class_name].append(detection)
        
        # Draw summary panel
        summary_height = len(class_detections) * 30 + 40
        overlay = frame.copy()
        panel_x = frame.shape[1] - 250
        cv2.rectangle(overlay, (panel_x, 10), (frame.shape[1] - 10, summary_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw summary title
        cv2.putText(frame, "Detection Summary:", (panel_x + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw class summaries
        for i, (class_name, items) in enumerate(class_detections.items()):
            y_position = 60 + i * 30
            color = self.get_color_for_class(class_name)
            count = len(items)
            avg_confidence = np.mean([d['confidence'] for d in items])
            
            summary_text = f"{class_name}: {count} ({avg_confidence:.2f})"
            cv2.putText(frame, summary_text, (panel_x + 10, y_position), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame and return annotated result"""
        # Perform inference
        results = self.model(frame)
        
        detections = []
        for result in results:
            for box in result.boxes:
                # Extract box information
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                
                # Filter low confidence detections
                if confidence < 0.25:
                    continue
                
                # Store detection info
                detections.append({
                    'box': [x1, y1, x2, y2],
                    'class': class_name,
                    'confidence': confidence
                })
                
                # Draw detection on frame
                self.draw_detection_info(frame, [x1, y1, x2, y2], class_name, confidence)
        
        # Draw statistics panel
        self.draw_statistics(frame, detections)
        
        # Draw detection summary
        self.draw_detection_summary(frame, detections)
        
        return frame
    
    def run_detection(self, source: int = 0) -> None:
        """Main method to run object detection on video source"""
        cap = cv2.VideoCapture(source)
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("=" * 50)
        print("YOLOv5su Object Detection Demo")
        print("=" * 50)
        print("Controls:")
        print("- Press 'q' to quit")
        print("- Press 's' to save current frame")
        print("- Press 'p' to pause/resume")
        print("=" * 50)
        
        paused = False
        
        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Process the frame
                processed_frame = self.process_frame(frame)
                
                # Display the result
                cv2.imshow('YOLOv5su Object Detection - Internship Demo', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"yolov5su_detection_{timestamp}.jpg"
                cv2.imwrite(filename, processed_frame)
                print(f"Frame saved as {filename}")
            elif key == ord('p'):
                paused = not paused
                status = "Paused" if paused else "Resumed"
                print(f"Detection {status}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Object detection stopped.")

def main():
    """Main function to run the object detector"""
    # Initialize detector with YOLOv5su model
    detector = ObjectDetector('yolov5su.pt')
    
    try:
        # Use webcam (0) or video file path
        detector.run_detection(source=0)
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the model file exists and camera is accessible")
        print("Available webcam indices: Try 0, 1, 2 if 0 doesn't work")

if __name__ == "__main__":
    main()