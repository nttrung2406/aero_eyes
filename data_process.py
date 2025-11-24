import os, cv2, json
from pathlib import Path
from tqdm import tqdm

def extract_annotated_frames(train_root, output_dir):
    """
    Convert the drone dataset (samples/, annotations/) into YOLO-style dataset.
    Args:
        train_root: root directory containing 'samples/' and 'annotations/'
        output_dir: directory to save 'images/' and 'labels/'
    """
    ann_path = Path(train_root) / "annotations" / "annotations.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"❌ Missing annotation file: {ann_path}")

    with open(ann_path) as f:
        anns = json.load(f)

    img_dir = Path(output_dir) / "images"
    lbl_dir = Path(output_dir) / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    items_root = Path(train_root) / "samples"

    for item in tqdm(anns, desc="Extracting annotated frames"):
        video_id = item["video_id"]
        detections = item.get("annotations", [])
        video_path = items_root / video_id / "drone_video.mp4"

        if not video_path.exists():
            print(f"Missing video: {video_path}")
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Cannot open {video_path}")
            continue

        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        for det in detections:
            for bbox in det.get("bboxes", []):
                frame_idx = bbox["frame"]
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                img_name = f"{video_id}_{frame_idx:06d}.jpg"
                img_path = img_dir / img_name
                label_path = lbl_dir / f"{video_id}_{frame_idx:06d}.txt"

                x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
                xc, yc = ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h
                bw, bh = (x2 - x1) / w, (y2 - y1) / h

                cv2.imwrite(str(img_path), frame)
                with open(label_path, "w") as f:
                    f.write(f"0 {xc} {yc} {bw} {bh}\n")

        cap.release()

    print(f"\nConversion complete → {output_dir}")
