import cv2
import matplotlib.pyplot as plt

video_path = r"E:\videos\L22\L22_V030.mp4"
k = (12 * 60 + 15) * 25   # bắt đầu từ frame 100
n = 20    # số frame muốn xem

cap = cv2.VideoCapture(video_path)

frames = []
for i in range(k, k + n):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if not ret:
        break
    # đổi sang RGB để matplotlib hiển thị đúng màu
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append((i, frame))

cap.release()

# hiển thị n frame trên lưới
plt.figure(figsize=(20, 10))
for idx, (frame_idx, frame) in enumerate(frames):
    plt.subplot(4, 5, idx+1)  # 4 hàng, 5 cột
    plt.imshow(frame)
    plt.title(f"Frame {frame_idx}")
    plt.axis("off")

plt.tight_layout()
plt.show()