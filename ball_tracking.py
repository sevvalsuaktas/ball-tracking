"""
Otonom top takibi: basketbol ve tenis topu için tek kod

Kullanım:
    python ball_tracking.py basketball.mp4 --ball basketball
    python ball_tracking.py tennisball.mp4 --ball tennis
    python ball_tracking.py basketball.mp4 --save ciktibasketball.mp4     # pencere açmadan videoya kaydeder
    python ball_tracking.py tennisball.mp4 --save ciktitennisball.mp4    

Pencerede 'q' tuşu çıkış yapar.
"""
import argparse
import cv2
import numpy as np

#  HSV (OpenCV): H 0-180, S 0-255, V 0-255
PROFILES = {
    'basketball': dict(
        ref_h=516,                  # piksel değerlerinin ayarlandığı video yüksekliği (otomatik ölçeklenir)
        strict=[((4, 80, 50), (15, 255, 255))],
        # Parlak ışığa yakın/bulanık karelerde top solar, ton pembeye kayar (H~165-180).
        # Gevşek aralık SADECE tahmin edilen konumun çevresinde kullanılır.
        loose=[((0, 35, 60), (15, 255, 255)), ((165, 35, 60), (180, 255, 255))],
        min_r=6, max_r=45,          # top yarıçapı sınırları (ref_h'de piksel)
        acq_min_r=13,               # yeni takip başlatmak için min yarıçap
        reacq_min_r=8,              # takip kaybolunca yeniden yakalama için min yarıçap
        min_area=20, loose_min_area=40,
        min_circ=0.55, min_fill=0.60,
        close_iter=3,
        gate=90,                    # tahmin edilen konuma max uzaklık (piksel)
        max_lost=8,                 # bu kadar kare bulunamazsa takibi bırak
        use_dist_acquire=True,      # el/deriyle birleşen topu mesafe dönüşümüyle yakala
    ),
    'tennis': dict(
        ref_h=1080,
        # Videodan ölçülen top: H~38-40, S~130-145, V~181-188 (parlak, doygun sarı-yeşil).
        # Üstteki yeşil file/perde H~41 ama S<100, V<100: S>=100 ve V>=120 ile elenir.
        strict=[((30, 100, 120), (48, 255, 255))],
        loose=[((25, 70, 100), (55, 255, 255))],
        min_r=10, max_r=90,
        acq_min_r=20,
        reacq_min_r=14,
        min_area=150, loose_min_area=200,
        min_circ=0.55, min_fill=0.60,
        close_iter=2,
        gate=220,                   # tenis topu hızlı: kapı daha geniş
        max_lost=8,
        use_dist_acquire=False,     # tenis topu deriyle birleşmez; sadece yuvarlak adaylar
    ),
}

CUT_MAD = 40     # küçültülmüş gri görüntüler arası ort. fark bunu aşarsa sahne kesmesi
CUT_CORR = 0.6   # veya renk histogramı korelasyonu bunun altına düşerse

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

def scale_profile(P, H):
    """Piksel cinsinden ayarları gerçek video yüksekliğine göre ölçekler."""
    s = H / float(P['ref_h'])
    Q = dict(P)
    for k in ('min_r', 'max_r', 'acq_min_r', 'reacq_min_r', 'gate'):
        Q[k] = P[k] * s
    for k in ('min_area', 'loose_min_area'):
        Q[k] = P[k] * s * s
    return Q

def color_mask(hsv, ranges):
    m = None
    for lo, hi in ranges:
        part = cv2.inRange(hsv, np.array(lo), np.array(hi))
        m = part if m is None else cv2.bitwise_or(m, part)
    return m

def clean_mask(m, close_iter):
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=close_iter)

def make_kalman():
    kf = cv2.KalmanFilter(4, 2)
    kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
    kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 2.0
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    return kf

def find_candidates(mask, P):
    """Maskeden 'top gibi' (yuvarlak, dolu, makul boyutlu) adayları çıkarır."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < P['min_area']:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        if not (P['min_r'] <= r <= P['max_r']):
            continue
        per = cv2.arcLength(c, True)
        circ = 4 * np.pi * area / (per * per + 1e-6)
        fill = area / (np.pi * r * r + 1e-6)
        if circ < P['min_circ'] or fill < P['min_fill']:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        if not (0.4 < bw / float(bh) < 2.5):
            continue
        out.append({'x': x, 'y': y, 'r': r, 'score': 0.5 * circ + 0.5 * fill,
                    'box': (bx, by, bw, bh)})
    return out

def _roi(shape, pred, R):
    px, py = int(pred[0]), int(pred[1])
    x0, y0 = max(px - R, 0), max(py - R, 0)
    x1, y1 = min(px + R, shape[1]), min(py + R, shape[0])
    return x0, y0, x1, y1

def find_by_distance(mask, pred, last_r, P):
    """Top kol/deriyle tek blob olduysa: tahmin çevresindeki en 'kalın' bölgeyi top say."""
    x0, y0, x1, y1 = _roi(mask.shape, pred, int(P['gate']))
    roi = mask[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    dist = cv2.distanceTransform(roi, cv2.DIST_L2, 5)
    _, mx, _, loc = cv2.minMaxLoc(dist)
    if mx < P['min_r'] * 0.8 or mx > P['max_r']:
        return None
    if last_r and abs(mx - last_r) / last_r > 0.5:
        return None
    x, y = loc[0] + x0, loc[1] + y0
    return {'x': float(x), 'y': float(y), 'r': mx, 'score': 0.3,
            'box': (int(x - mx), int(y - mx), int(2 * mx), int(2 * mx))}

def acquire_by_distance(mask, ref_r, P):
    """Düzgün yuvarlak aday yoksa: maskedeki en kalın bölgeyi al (kenara değenler hariç)."""
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, mx, _, loc = cv2.minMaxLoc(dist)
    if mx < P['acq_min_r'] or mx > P['max_r']:
        return None
    if ref_r and not (0.6 < mx / ref_r < 1.6):
        return None
    x, y = loc
    m = int(mx) + 4
    if x < m or y < m or x > mask.shape[1] - m or y > mask.shape[0] - m:
        return None          # görüntü kenarına değen blok = büyük ihtimalle kol/bacak
    return {'x': float(x), 'y': float(y), 'r': mx, 'score': 0.3,
            'box': (int(x - mx), int(y - mx), int(2 * mx), int(2 * mx))}

def find_loose(hsv, pred, last_r, P):
    """Sıkı maske bulamadıysa tahmin çevresinde gevşek renk maskesiyle ara."""
    x0, y0, x1, y1 = _roi(hsv.shape, pred, int(P['gate']))
    roi = hsv[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    m = clean_mask(color_mask(roi, P['loose']), P['close_iter'])
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_cost = None, 1e9
    for c in cnts:
        area = cv2.contourArea(c)
        if area < P['loose_min_area']:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        if not (P['min_r'] <= r <= P['max_r']):
            continue
        if last_r and not (0.6 < r / last_r < 1.5):   # önceki top boyutuna yakın olmalı
            continue
        fill = area / (np.pi * r * r + 1e-6)
        if fill < 0.5:
            continue
        gx, gy = x + x0, y + y0
        cost = np.hypot(gx - pred[0], gy - pred[1]) - 20 * fill
        if cost < best_cost:
            bx, by, bw, bh = cv2.boundingRect(c)
            best_cost = cost
            best = {'x': gx, 'y': gy, 'r': r, 'score': 0.4, 'src': 'loose',
                    'box': (bx + x0, by + y0, bw, bh)}
    return best

def auto_profile(cap, n_samples=30):
    """Videodan eşit aralıklı kareler alıp her profilin 'top gibi' aday bulma başarısını ölçer."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        return 'basketball', {}
    idx = np.linspace(0, total - 1, min(n_samples, total)).astype(int)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stats = {name: [0, 0.0] for name in PROFILES}      # [aday bulunan kare sayısı, skor toplamı]
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok:
            continue
        hsv = cv2.cvtColor(cv2.GaussianBlur(f, (5, 5), 0), cv2.COLOR_BGR2HSV)
        for name, P0 in PROFILES.items():
            P = scale_profile(P0, H)
            m = clean_mask(color_mask(hsv, P['strict']), P['close_iter'])
            good = [c for c in find_candidates(m, P) if c['r'] >= P['acq_min_r'] and c['score'] > 0.65]
            if good:
                stats[name][0] += 1
                stats[name][1] += max(c['score'] for c in good)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    best = max(stats, key=lambda k: (stats[k][0], stats[k][1]))
    if stats[best][0] == 0:
        best = 'basketball'
    return best, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video')
    ap.add_argument('--ball', choices=['auto', 'basketball', 'tennis'], default='auto')
    ap.add_argument('--save', default=None, help='çıktıyı bu dosyaya yaz (pencere açmaz)')
    ap.add_argument('--tolerance', type=int, default=None, help='"MERKEZDE" bölgesi (piksel); varsayılan: genişliğin %%9.5\'i')
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f'Video açılamadı: {args.video}')

    name = args.ball
    if name == 'auto':
        name, stats = auto_profile(cap)
        print('Otomatik profil seçimi:', name, '| (aday bulunan kare, skor):',
              {k: (v[0], round(v[1], 1)) for k, v in stats.items()})
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    P = scale_profile(PROFILES[name], H)
    print(f'Profil: {name}')

    delay = int(1000 / fps)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    CX, CY = W // 2, H // 2
    # "MERKEZDE" bölgesi: verilmezse kare genişliğinin %9.5'i (526 px'te ~50 px)
    ui = max(1.0, H / 540.0)
    ti = lambda v: max(1, int(round(v * ui)))      # kalınlık/yarıçap ölçekleyici
    TOL = args.tolerance if args.tolerance is not None else int(round(0.095 * W))
    if not args.save:
        for wn in ("Otonom Top Takibi", "Arka Plan Maskesi"):
            cv2.namedWindow(wn, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(wn, 960, int(960 * H / W))

    writer = None
    if args.save:
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

    kf = None
    tracking = False
    lost = 0
    last_r = None
    prev_hist = prev_gray = None
    last_print_time = -1000

    def global_best(cs):
        return max(cs, key=lambda c: c['score'] + 0.02 * c['r']) if cs else None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

        # --- Sahne kesmesi (montaj) tespiti: kesme olunca takibi sıfırla ---
        small = cv2.resize(frame, (64, 64))
        gray_s = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32, 32)).astype(np.float32)
        hist = cv2.calcHist([cv2.cvtColor(small, cv2.COLOR_BGR2HSV)], [0, 1], None,
                            [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        if prev_hist is not None:
            mad = float(np.abs(gray_s - prev_gray).mean())
            corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if mad > CUT_MAD or corr < CUT_CORR:
                tracking, kf, lost, last_r = False, None, 0, None
        prev_hist, prev_gray = hist, gray_s

        # --- Renk maskesi ---
        hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
        mask = clean_mask(color_mask(hsv, P['strict']), P['close_iter'])
        cands = find_candidates(mask, P)

        # --- Kalman tahmini ---
        pred = None
        if tracking:
            p = kf.predict()
            pred = (float(p[0, 0]), float(p[1, 0]))

        # --- En iyi adayı seç ---
        best = None
        if tracking:
            def cost(c):
                d = np.hypot(c['x'] - pred[0], c['y'] - pred[1])
                size_pen = abs(c['r'] - last_r) / max(last_r, 1)
                return d + 40 * size_pen - 30 * c['score']
            near = [c for c in cands if np.hypot(c['x'] - pred[0], c['y'] - pred[1]) < P['gate']]
            if near:
                best = min(near, key=cost)
            else:
                # kapı içinde yuvarlak aday yok: birleşik blob -> gevşek renk maskesi
                if P['use_dist_acquire']:
                    best = find_by_distance(mask, pred, last_r, P)
                best = best or find_loose(hsv, pred, last_r, P)
                # birkaç kare bulunamadıysa (ör. sahne değişti) her yerden yeniden yakala
                if best is None and lost >= 2:
                    strong = [c for c in cands if c['r'] >= P['reacq_min_r'] and c['score'] > 0.65]
                    best = global_best(strong)
                    if best is None and lost >= 5 and P['use_dist_acquire']:
                        best = acquire_by_distance(mask, last_r, P)
                    if best is not None:
                        tracking = False   # yeni takip başlat
        else:
            strong = [c for c in cands if c['r'] >= P['acq_min_r'] and c['score'] > 0.6]
            best = global_best(strong)
            if best is None and P['use_dist_acquire']:
                best = acquire_by_distance(mask, None, P)

        cv2.circle(frame, (CX, CY), ti(5), (0, 255, 255), -1)

        ball = None
        if best is not None:
            new_track = not tracking
            if new_track:
                kf = make_kalman()
                kf.statePost = np.array([[best['x']], [best['y']], [0], [0]], np.float32)
                tracking = True
            else:
                kf.correct(np.array([[best['x']], [best['y']]], np.float32))
            lost = 0
            if new_track or last_r is None:
                last_r = best['r']
            elif best.get('src') != 'loose':       # gevşek tespit top boyutunu bozmasın
                last_r = 0.7 * last_r + 0.3 * best['r']
            ball = (int(best['x']), int(best['y']), best['box'], True)
        elif tracking:
            lost += 1
            left_view = not (0 <= pred[0] <= W and 0 <= pred[1] <= H)   # merkez kadraj dışına çıktı
            if lost > P['max_lost'] or left_view:
                tracking, kf = False, None
            else:
                # kısa süreli kayıp: tahmin edilen konumu kullan (kesikli kutu)
                kf.statePost[2:] *= 0.7
                r = int(last_r)
                ball = (int(pred[0]), int(pred[1]),
                        (int(pred[0] - r), int(pred[1] - r), 2 * r, 2 * r), False)

        if ball is not None:
            bx_c, by_c, (x, y, w, h), detected = ball
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0) if detected else (0, 200, 255), ti(2))
            cv2.circle(frame, (bx_c, by_c), ti(5), (0, 0, 255), -1)
            cv2.arrowedLine(frame, (CX, CY), (bx_c, by_c), (255, 0, 0), ti(3), tipLength=0.05)

            dx = bx_c - CX
            if dx < -TOL:
                command, color = "SOLA DON", (0, 165, 255)
            elif dx > TOL:
                command, color = "SAGA DON", (0, 165, 255)
            else:
                command, color = "MERKEZDE", (0, 255, 0)

            cv2.putText(frame, f"KOMUT: {command}", (ti(20), ti(40)), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * ui, color, ti(2))
            cv2.putText(frame, f"dx={dx}  dy={by_c - CY}", (ti(20), ti(70)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6 * ui, (255, 255, 255), ti(2))

            if t_ms - last_print_time >= 100:
                print(f"Zaman: {int(t_ms)}ms | Top Merkezi: X={bx_c}, Y={by_c} | dx={dx} | KOMUT: {command}")
                last_print_time = t_ms
        else:
            cv2.putText(frame, "TOP YOK", (ti(20), ti(40)), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * ui, (0, 0, 255), ti(2))

        cv2.putText(frame, f"profil: {name}", (ti(10), H - ti(10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * ui, (200, 200, 200), ti(1))

        if writer:
            writer.write(frame)
        else:
            cv2.imshow("Otonom Top Takibi", frame)
            cv2.imshow("Arka Plan Maskesi", mask)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()