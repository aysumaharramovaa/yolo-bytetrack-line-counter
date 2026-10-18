import argparse                                
import time                                    
from collections import defaultdict, deque     


import cv2                                  
import numpy as np                            
from ultralytics import YOLO                  


def side_of_line(p, a, b):
  
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def crossed_segment(prev_p, cur_p, a, b):

    o1 = side_of_line(prev_p, a, b)    
    o2 = side_of_line(cur_p, a, b)     
    o3 = side_of_line(a, prev_p, cur_p)  
    o4 = side_of_line(b, prev_p, cur_p)  
    return ((o1 > 0) != (o2 > 0)) and ((o3 > 0) != (o4 > 0))

def color_for_id(track_id):
    """
    Hər ID üçün həmişə eyni rəng qaytarır.
    ID-ni "seed" kimi veririk, ona görə #7 hər kadrda eyni rəngdə görünür.
    Rəng 60-255 aralığındadır ki, çox tünd (görünməyən) olmasın.
    """
    rng = np.random.RandomState(track_id)
    return tuple(int(c) for c in rng.randint(60, 255, 3))


def draw_counter_panel(frame, counts, fps):

    # counts -> {"car": {"ab": 12, "ba": 3}, "bus": {...}, ...}

    rows = list(counts.items())
    line_h = 26                    
    x0, y0 = 10, 10                
    panel_w = 270
    n_lines = len(rows) + 3    
    panel_h = 16 + n_lines * line_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    x_name, x_ab, x_ba = x0 + 12, x0 + 140, x0 + 205 

    def put(text, x, y, color=(255, 255, 255), scale=0.6):
        cv2.putText(frame, text, (x, y), font, scale, color, 1, cv2.LINE_AA)

    y = y0 + 26
    put("class", x_name, y, (200, 200, 200))
    put("A->B", x_ab, y, (200, 200, 200))
    put("B->A", x_ba, y, (200, 200, 200))

    total_ab = total_ba = 0
    for name, c in rows:
        y += line_h
        put(name, x_name, y)
        put(str(c["ab"]), x_ab, y, (0, 255, 0))
        put(str(c["ba"]), x_ba, y, (0, 255, 0))
        total_ab += c["ab"]
        total_ba += c["ba"]

    y += line_h
    put("TOTAL", x_name, y, (0, 255, 255))
    put(str(total_ab), x_ab, y, (0, 255, 255))
    put(str(total_ba), x_ba, y, (0, 255, 255))

    y += line_h
    put(f"FPS: {fps:.1f}", x_name, y, (180, 180, 180), 0.5)


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO + ByteTrack xətt sayğacı")

    parser.add_argument("--source", default="0",
                        help="0 = veb kamera, ya da video faylının yolu")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="YOLO modeli (n=sürətli, s/m/l/x=daha dəqiq)")
    parser.add_argument("--classes", type=int, nargs="+", default=[0],
                        help="COCO sinif nömrələri (boşluqla). 0=insan, 2=maşın, "
                             "3=motosiklet, 5=avtobus, 7=yük maşını")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Minimum əminlik (confidence) həddi")
    parser.add_argument("--line", type=int, nargs=4, default=None,
                        metavar=("X1", "Y1", "X2", "Y2"),
                        help="Sayğac xəttinin koordinatları (piksel). Verilməsə, ortada üfüqi xətt.")
    parser.add_argument("--cooldown", type=int, default=15,
                        help="Eyni ID-nin təkrar sayılması üçün minimum kadr fasiləsi")
    parser.add_argument("--forget", type=int, default=150,
                        help="Bu qədər kadr görünməyən ID yaddaşdan silinir")
    parser.add_argument("--save", default=None,
                        help="Nəticə videosunu bu fayla yaz")
    parser.add_argument("--no-show", action="store_true",
                        help="Pəncərə açma (serverdə faydalıdır)")
    return parser.parse_args()


def main():
    args = parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    model = YOLO(args.model)

    for c in args.classes:
        if c not in model.names:
            raise SystemExit(f"Yanlış sinif nömrəsi: {c}. Mövcud: 0-{len(model.names) - 1}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Mənbə açılmadı: {args.source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30    
    if args.line:
        x1, y1, x2, y2 = args.line
        line_a, line_b = (x1, y1), (x2, y2)
    else:
        line_a, line_b = (0, height // 2), (width, height // 2)

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, src_fps, (width, height))

    trails = defaultdict(lambda: deque(maxlen=30))
    last_point = {}   
    last_seen = {}        
    last_counted = {}     

    counts = defaultdict(
        lambda: {"ab": 0, "ba": 0},
        {model.names[c]: {"ab": 0, "ba": 0} for c in args.classes},
    )

    frame_idx = 0  
    prev_time = time.time() 
    while True:
        ok, frame = cap.read()
        if not ok:
            break         
        frame_idx += 1

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=args.classes,
            conf=args.conf,
            verbose=False,
        )[0]             

        boxes = results.boxes

        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()        
            ids = boxes.id.int().cpu().tolist()   
            classes = boxes.cls.int().cpu().tolist() 
            confs = boxes.conf.cpu().tolist()    
            for (bx1, by1, bx2, by2), tid, cls_id, conf in zip(xyxy, ids, classes, confs):
                bx1, by1, bx2, by2 = int(bx1), int(by1), int(bx2), int(by2)
                color = color_for_id(tid)
                cls_name = model.names[cls_id] 
                foot = (int((bx1 + bx2) / 2), by2)
                trails[tid].append(foot)   
                last_seen[tid] = frame_idx  
                if tid in last_point:
                    if crossed_segment(last_point[tid], foot, line_a, line_b):
                        if frame_idx - last_counted.get(tid, -10**9) >= args.cooldown:
                            if side_of_line(foot, line_a, line_b) > 0:
                                counts[cls_name]["ab"] += 1
                            else:
                                counts[cls_name]["ba"] += 1
                            last_counted[tid] = frame_idx
                last_point[tid] = foot    


                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)

                label = f"{cls_name} #{tid} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (bx1, by1 - th - 8), (bx1 + tw + 4, by1), color, -1)
                cv2.putText(frame, label, (bx1 + 2, by1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                pts = np.array(trails[tid], dtype=np.int32).reshape(-1, 1, 2)
                if len(pts) > 1:
                    cv2.polylines(frame, [pts], False, color, 2)
                cv2.circle(frame, foot, 4, color, -1)  

        if frame_idx % 60 == 0:
            stale = [t for t, seen in last_seen.items() if frame_idx - seen > args.forget]
            for t in stale:
                last_seen.pop(t, None)
                last_point.pop(t, None)
                last_counted.pop(t, None)
                trails.pop(t, None)

        cv2.line(frame, line_a, line_b, (0, 255, 255), 2)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)  
        prev_time = now

        draw_counter_panel(frame, counts, fps)

        if writer:
            writer.write(frame)
        if not args.no_show:
            cv2.imshow("CV Tracker (q = cixis)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print("Yekun nəticə:")
    for name, c in counts.items():
        print(f"  {name:<12} A->B = {c['ab']:<4} B->A = {c['ba']}")


if __name__ == "__main__":
    main()