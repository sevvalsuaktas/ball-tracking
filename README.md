# Tenis Topu Takibi 🎾

Bu proje, TEKNOFEST görevi kapsamında OpenCV kullanılarak HSV renk uzayında hareketli bir tenis topunun tespit edilmesi ve takip edilmesi için geliştirilmiştir. Algoritma, topun merkez koordinatlarını (X, Y) hesaplar ve 100 milisaniyede bir terminale yazdırır.

## Kurulum ve Gereksinimler

Kodu çalıştırabilmek için bilgisayarınızda Python yüklü olmalıdır. Gerekli kütüphaneleri kurmak için terminalde şu komutu çalıştırın:

```bash
pip install opencv-python numpy

Nasıl Çalıştırılır?
Video dosyaları yüksek boyutlu olduğu için bu depoya eklenmemiştir. Kodu kendi bilgisayarınızda denemek için aşağıdaki adımları izleyin:

Test edeceğiniz tenis topu videosunu bu proje klasörünün içine indirin.

Videonun adını tam olarak video.mp4 yapın (veya kod içindeki cv2.VideoCapture('video.mp4') kısmını kendi videonuzun adıyla güncelleyin).

Terminalde kodu çalıştırın:
python main.py
(Eğer dosyanın adını tenisball.py yaptıysanız komutu python tenisball.py olarak değiştirin)