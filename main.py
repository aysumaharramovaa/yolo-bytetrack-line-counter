import argparse
import time
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO



def side_of_line(p, a, b):
    """
        > 0  -> bir tərəf
        < 0  -> o biri tərəf
        = 0  -> düz xəttin üstündə
    """
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def color_for_id(track_id):
    rng = np.random.RandomState(track_id)
    return tuple(int(c) for c in rng.randint(60, 255, 3))


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO + ByteTrack xətt sayğacı")
    parser.add_argument("--source", default="0",
                        help="0 = veb kamera, ya da video faylının yolu")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="YOLO modeli (n=sürətli, s/m/l/x=daha dəqiq)")
    parser.add_argument("--classes", type=int, nargs="+", default=[0],
                        help="COCO sinif nömrələri. 0=insan, 2=maşın, 3=motosiklet")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Minimum əminlik (confidence) həddi")
    parser.add_argument("--line", type=int, nargs=4, default=None,
                        metavar=("X1", "Y1", "X2", "Y2"),
                        help="Sayğac xəttinin koordinatları (piksel). Verilməsə, ortada üfüqi xətt.")
    parser.add_argument("--save", default=None, help="Nəticə videosunu bu fayla yaz")
    parser.add_argument("--no-show", action="store_true", help="Pəncərə açma (serverdə faydalıdır)")
    return parser.parse_args()


def main():
    args = parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    model = YOLO(args.model)

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
    last_side = {}                                 
    count_ab = 0                                 
    count_ba = 0                                    

    prev_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break  
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

                foot = (int((bx1 + bx2) / 2), by2)
                trails[tid].append(foot)

                side = side_of_line(foot, line_a, line_b)
                sign = 1 if side > 0 else (-1 if side < 0 else 0)

                if sign != 0:
                    prev = last_side.get(tid)
                    if prev is not None and prev != sign:
                        if sign > 0:
                            count_ab += 1
                        else:
                            count_ba += 1
                    last_side[tid] = sign

                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
                label = f"{model.names[cls_id]} #{tid} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (bx1, by1 - th - 8), (bx1 + tw + 4, by1), color, -1)
                cv2.putText(frame, label, (bx1 + 2, by1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                pts = np.array(trails[tid], dtype=np.int32).reshape(-1, 1, 2)
                if len(pts) > 1:
                    cv2.polylines(frame, [pts], False, color, 2)
                cv2.circle(frame, foot, 4, color, -1)

        cv2.line(frame, line_a, line_b, (0, 255, 255), 2)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        cv2.putText(frame, f"A->B: {count_ab}   B->A: {count_ba}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

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
    print(f"Yekun: A->B = {count_ab}, B->A = {count_ba}")


if __name__ == "__main__":
    main()
