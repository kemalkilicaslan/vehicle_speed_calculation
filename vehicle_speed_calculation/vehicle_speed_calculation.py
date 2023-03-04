# Vehicle Speed Calculation
# Araç Hızı Hesabı

import cv2
import numpy as np

# YOLOv4 nesne algılama modelini yüklüyoruz.
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

# Modelin algılayabileceği sınıfları tanımlıyoruz.
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Ağ için giriş ve çıkış katmanlarını ayarlayalım.
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in [net.getUnconnectedOutLayers()]]

# Video yakalama parametrelerini ayarları,
video_path = "video.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
scale = 1.5  # videonun ölçeğine uyması için bu değeri ayarlama kısmı,

# Araç takibi için değişkenleri başlatma kısmı,
prev_frame = None
prev_bboxes = None
prev_centroids = None

# Video işleme döngüsünü başlatıyoruz.
while True:
    # Videodan kare okuma;
    ret, frame = cap.read()
    if not ret:
        break

    # Daha hızlı işleme için çerçeveyi yeniden boyutlandırma;
    height, width = frame.shape[:2]
    new_width = int(width / scale)
    new_height = int(height / scale)
    resized_frame = cv2.resize(frame, (new_width, new_height))

    # Yeniden boyutlandırılmış çerçevede nesne algılama gerçekleştirme;
    blob = cv2.dnn.blobFromImage(resized_frame, 1 / 255, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Tespit sonuçlarını ayrıştırma ve araçlar için filtreleleme;
    bboxes = []
    centroids = []
    confidences = []
    class_ids = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "car":
                center_x = int(detection[0] * new_width)
                center_y = int(detection[1] * new_height)
                w = int(detection[2] * new_width)
                h = int(detection[3] * new_height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                bboxes.append([x, y, w, h])
                centroids.append((center_x, center_y))
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Araç takibi ve hız tahmini yapma;
    if prev_frame is not None and len(bboxes) > 0:
        curr_centroids = np.array(centroids)
        prev_centroids = np.resize(prev_centroids, curr_centroids.shape)
        dists = np.linalg.norm(curr_centroids - prev_centroids, axis=1)
        speeds = dists * fps / scale  # hızı saniyedeki piksel cinsinden hesaplama kısmı,


        for i in range(len(bboxes)):
            x, y, w, h = bboxes[i]
            speed = speeds[i]
            # sınırlayıcı kutu çizip ve çerçevedeki metni hızlandırma kısmı,
            cv2.rectangle(frame, (int(x * scale), int(y * scale)), (int((x + w) * scale), int((y + h) * scale)),
                          (0, 255, 0), 2)
            cv2.putText(frame, f"{speed:.1f} px/s", (int(x * scale), int((y - 10))), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2)

    # İşlenen çerçeveyi gösteriyoruz.
    cv2.imshow("Frame", frame)

    # Çıkmak için kullanıcı girişini kontrol ediyoruz.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # Bir sonraki yineleme için önceki çerçeveyi ve ağırlık merkezlerini ayarlıyoruz.
    prev_frame = resized_frame
    prev_centroids = centroids

# Kodu sonlandırıyoruz.
cap.release()
cv2.destroyAllWindows()