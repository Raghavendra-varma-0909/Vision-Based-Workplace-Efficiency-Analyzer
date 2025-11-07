import cv2

cap = cv2.VideoCapture(0)  # Try camera 0
if not cap.isOpened():
    print("❌ Camera 0 not detected.")
else:
    print("✅ Camera 0 opened successfully.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Can't read frame.")
            break
        cv2.imshow("Webcam Test - Press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
