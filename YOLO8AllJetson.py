import time
import cv2
import torch
import psutil
from ultralytics import YOLO

# โหลดโมเดล YOLOv8 และย้ายไปยัง GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8n.pt").to(device)

# เปิดไฟล์วิดีโอ
video_path = "Demo_Video.mp4"
cap = cv2.VideoCapture(video_path)

# ตรวจสอบว่าเปิดไฟล์วิดีโอได้หรือไม่
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# เตรียมตัวแปรสำหรับวัดค่า
frame_count = 0
start_time = time.time()

# ตัวแปรสำหรับเก็บผลลัพธ์เพื่อวัดความแม่นยำ
all_precisions = []
all_recalls = []
all_f1s = []

# อ่านและประมวลผลเฟรมวิดีโอทีละเฟรม
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # วัดเวลาเริ่มต้นสำหรับเฟรมนี้
    frame_start_time = time.time()

    # รันโมเดลบนเฟรม
    results = model(frame)
    predictions = results.pred[0].tolist()  # แปลงผลลัพธ์เป็น list ของ bounding boxes

    # โหลด ground truth สำหรับเฟรมนี้
    ground_truths = load_ground_truth(frame_count)

    # คำนวณ precision, recall, และ f1 score
    precision, recall, f1 = calculate_metrics(predictions, ground_truths)
    all_precisions.append(precision)
    all_recalls.append(recall)
    all_f1s.append(f1)

    # วัดเวลาเสร็จสิ้นสำหรับเฟรมนี้
    frame_end_time = time.time()
    frame_time = frame_end_time - frame_start_time

    # เพิ่มจำนวนเฟรมที่ประมวลผล
    frame_count += 1

    # วัดการใช้หน่วยความจำ
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_usage = memory_info.rss / 1024 / 1024  # แปลงเป็น MB

    # แสดงข้อมูลเฟรมและการใช้หน่วยความจำ
    print(f"Frame: {frame_count}, Time per frame: {frame_time:.4f} seconds")
    print(f"Memory usage: {memory_usage:.4f} MB")

# คำนวณเวลาและ FPS
end_time = time.time()
total_time = end_time - start_time
fps = frame_count / total_time

# คำนวณความแม่นยำเฉลี่ย
avg_precision = sum(all_precisions) / len(all_precisions)
avg_recall = sum(all_recalls) / len(all_recalls)
avg_f1 = sum(all_f1s) / len(all_f1s)

print(f"Total frames processed: {frame_count}")
print(f"Total time: {total_time:.4f} seconds")
print(f"Throughput (FPS): {fps:.2f}")
print(f"Average Precision: {avg_precision:.4f}")
print(f"Average Recall: {avg_recall:.4f}")
print(f"Average F1 Score: {avg_f1:.4f}")

# ปิดวิดีโอและหน้าต่างทั้งหมด
cap.release()
cv2.destroyAllWindows()
