import cv2
import numpy as np

def nothing(x):
    pass

# Ayar penceresi ve kızakları oluştur
cv2.namedWindow("Ayarlar", cv2.WINDOW_NORMAL)
cv2.createTrackbar("H Min", "Ayarlar", 0, 179, nothing)
cv2.createTrackbar("H Max", "Ayarlar", 25, 179, nothing)
cv2.createTrackbar("S Min", "Ayarlar", 50, 255, nothing)
cv2.createTrackbar("S Max", "Ayarlar", 255, 255, nothing)
cv2.createTrackbar("V Min", "Ayarlar", 50, 255, nothing)
cv2.createTrackbar("V Max", "Ayarlar", 255, 255, nothing)

cap = cv2.VideoCapture('basketball.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        # Video bitince başa sar (Sürekli döngü)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # Ekranlara sığması için boyutu küçült
    frame = cv2.resize(frame, (640, 360))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Kızaklardan o anki değerleri oku
    h_min = cv2.getTrackbarPos("H Min", "Ayarlar")
    h_max = cv2.getTrackbarPos("H Max", "Ayarlar")
    s_min = cv2.getTrackbarPos("S Min", "Ayarlar")
    s_max = cv2.getTrackbarPos("S Max", "Ayarlar")
    v_min = cv2.getTrackbarPos("V Min", "Ayarlar")
    v_max = cv2.getTrackbarPos("V Max", "Ayarlar")

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    # Maskeyi oluştur
    mask = cv2.inRange(hsv, lower, upper)

    cv2.imshow("Orijinal Video", frame)
    cv2.imshow("Maske (Siyah-Beyaz)", mask)

    key = cv2.waitKey(50) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        # 'p' tuşuna basarak videoyu duraklat, değerleri daha rahat ayarla
        cv2.waitKey(-1)

cap.release()
cv2.destroyAllWindows()