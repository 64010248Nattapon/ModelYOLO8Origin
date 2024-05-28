import time
import cv2
from ultralytics import YOLO
import torch

# โหลดโมเดล YOLOv8 และย้ายไปยัง GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('yolov8n.pt').to(device)

# เปิดไฟล์วิดีโอ
video_path = 'path/to/your/video.mp4'
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

    # วัดเวลาเริ่มต้นสำหรับเฟรมนี้
    frame_start_time = time.time()

    # รันโมเดลบนเฟรม
    results = model(frame)

    # วัดเวลาเสร็จสิ้นสำหรับเฟรมนี้
    frame_end_time = time.time()
    frame_time = frame_end_time - frame_start_time

    # เพิ่มจำนวนเฟรมที่ประมวลผล
    frame_count += 1

    # แสดงข้อมูลเฟรม
    print(f"Frame: {frame_count}, Time per frame: {frame_time:.4f} seconds")

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
print(f"Total frames processed: {frame_count}")
print(f"Total time: {total_time:.4f} seconds")
print(f"Throughput (FPS): {fps:.2f}")

# ปิดวิดีโอและหน้าต่างทั้งหมด
cap.release()
cv2.destroyAllWindows()
