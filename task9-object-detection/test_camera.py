import cv2

def test_camera():
    print("Testing camera...")
    
    # Try different camera indices
    for i in range(3):
        print(f"Testing camera index {i}...")
        cap = cv2.VideoCapture(i)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✅ Camera {i} is working! Frame shape: {frame.shape}")
                cap.release()
                return i
            else:
                print(f"❌ Camera {i} opens but can't read frames")
                cap.release()
        else:
            print(f"❌ Camera {i} not accessible")
    
    print("No working camera found!")
    return None

if __name__ == "__main__":
    working_camera = test_camera()
    if working_camera is not None:
        print(f"Use camera index: {working_camera}")
    else:
        print("Please check your camera connection and permissions")