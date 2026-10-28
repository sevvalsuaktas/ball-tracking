# Otonom Top Takibi ve Manevra Yönlendirmesi 🎾🏀

Bu proje, TEKNOFEST otonom sistemler görevi kapsamında OpenCV kullanılarak geliştirilmiştir. Algoritma, hareketli tenis ve basketbol toplarını HSV renk uzayında tespit eder, kameranın tam merkezi ile hedefin merkezi arasına bir yönlendirme vektörü çizer ve otonom araçlar için dinamik manevra komutları üretir.

## Proje Özellikleri
* **Çoklu Hedef Takibi:** Ortak bir maskeleme mantığıyla tek kod üzerinden hem tenis (sarı/yeşil) hem de basketbol (turuncu/kahverengi) toplarını eşzamanlı izleme.
* **Otonom Yönlendirme Vektörü:** Kameranın merkez noktasını (X, Y) referans alarak hedefe çizgi çekme ve 50 piksellik tolerans payıyla yönlendirme komutları (`SOLA DON`, `SAGA DON`, `MERKEZDE`) oluşturma.
* **Canlı HSV Kalibrasyonu:** Farklı saha koşullarında, ortam ışığında ve hareket bulanıklığında (motion blur) renk eşiklerini anlık olarak ayarlayabilmek için sisteme entegre edilen `calibration.py` aracı.

## Kurulum ve Gereksinimler

Kodu çalıştırabilmek için bilgisayarınızda Python yüklü olmalıdır. Gerekli kütüphaneleri kurmak için terminalde şu komutu çalıştırın:

```bash
pip install opencv-python numpy