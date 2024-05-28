import time
import cv2
from ultralytics import YOLO
import torch

# โหลดโมเดล YOLOv8 และย้ายไปยัง GPU
model = YOLO('yolov8n.pt').to('cuda' if torch.cuda.is_available() else 'cpu')

# เปิดไฟล์วิดีโอ
video_path = 'video.mp4'
cap = cv2.VideoCapture(video_path)

# ตรวจสอบว่าเปิดไฟล์วิดีโอได้หรือไม่
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# เตรียมตัวแปรสำหรับวัดค่า
frame_count = 0
start_time = time.time()

# อ่านและประมวลผลเฟรมวิดีโอทีละเฟรม
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # รันโมเดลบนเฟรม
    results = model(frame)

    # เก็บค่าความแม่นยำ (คุณอาจต้องมีโค้ดสำหรับการเปรียบเทียบกับ ground truth ที่ทำ annotation ไว้)
    # เช่น precision, recall, F1-score

    # เพิ่มจำนวนเฟรมที่ประมวลผล
    frame_count += 1

    # แสดงผลลัพธ์ (ถ้าต้องการ)
    # results.show()
    # cv2.imshow('Frame', frame)

    # กด q เพื่อออกจากการแสดงผล (ถ้าคุณใช้ cv2.imshow)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# คำนวณเวลาและ FPS
end_time = time.time()
total_time = end_time - start_time
fps = frame_count / total_time
print(f"Throughput (FPS): {fps}")

# ปิดวิดีโอและหน้าต่างทั้งหมด
cap.release()
cv2.destroyAllWindows()
