import cv2
import numpy as np

cap = cv2.VideoCapture('video.mp4')

# Videonun orijinal FPS değerini al
fps = cap.get(cv2.CAP_PROP_FPS)
# Videonun normal oynatma hızını hesapla
delay = int(1000 / fps) if fps > 0 else 30

# Tenis topu için HSV renk aralıkları
lower_color = np.array([28, 80, 80])
upper_color = np.array([45, 255, 255])

# 100 ms kontrolü için zaman tutucu değişken
last_print_time = 0

cv2.namedWindow("Tenis Topu Takibi", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tenis Topu Takibi", 960, 540)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break 

    # Videonun o anki milisaniye cinsinden oynatma süresi
    current_time = cap.get(cv2.CAP_PROP_POS_MSEC)

    # İşlemler için görüntüyü HSV'ye çevir 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    # Gürültüleri temizle
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=6)
    
    # Koordinatları bul
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        
        if w > 15 and h > 15:
            # En/boy oranını hesapla
            aspect_ratio = float(w) / h
            
            # Eğer şekil kareye yakınsa (0.7 ile 1.3 arasındaysa) top olarak kabul et
            if 0.7 < aspect_ratio < 1.3:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                center_x = x + w // 2
                center_y = y + h // 2
                
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"Merkez: ({center_x}, {center_y})", (x, y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                if current_time - last_print_time >= 100:
                    print(f"Zaman: {int(current_time)}ms - Topun Merkezi: X={center_x}, Y={center_y}")
                    last_print_time = current_time
    
    # üzerine çizim yapılmış orijinal renkli videoyu göster
    cv2.imshow("Tenis Topu Takibi", frame)
    
    # Videoyu orijinal hızına uygun bekleme süresiyle oynat
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()