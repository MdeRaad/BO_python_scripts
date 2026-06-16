"""
process_videos.py
=================
Verwerkt alle video's in de map videos/ en schrijft per video een CSV-bestand
naar data_OpenFace/ met per frame:
 
* gezichtsdetectie
* emotieherkenning
* blikrichting
* Action Units (AU01 t/m AU26 — intensiteit)
 
Verwachte mapstructuur (BO_PYTHON_SCRIPTS/ is de projectroot)::
 
    BO_PYTHON_SCRIPTS/
    ├── data_OpenFace/               ← uitvoer-CSV's komen hier terecht (wordt aangemaakt)
    ├── OpenFace/
    |   ├── OpenFace-3.0/                ← OpenFace 3.0 codebase (git submodule)
    │   └── proces_videos.py         ← dit script
    └── videos/                      ← invoervideo's
 
Afhankelijkheden:
    OpenCV, NumPy, PyTorch, OpenFace 3.0

"""

import cv2
import csv
import os
import sys
import tempfile
import torch
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
OPENFACE_DIR = SCRIPT_DIR / "OpenFace-3.0"

# OpenFace laadt intern weights via "./weights/..." dus chdir naar OpenFace-3.0
os.chdir(OPENFACE_DIR)
if str(OPENFACE_DIR) not in sys.path:
    sys.path.insert(0, str(OPENFACE_DIR))

from openface.face_detection import FaceDetector
from openface.multitask_model import MultitaskPredictor

# ── Config ────────────────────────────────────────────────────────────────────

VIDEO_DIR   = PROJECT_DIR / "videos"
OUTPUT_DIR  = PROJECT_DIR / "data_OpenFace"
FACE_MODEL  = str(OPENFACE_DIR / "weights" / "Alignment_RetinaFace.pth")
MULTI_MODEL = str(OPENFACE_DIR / "weights" / "MTL_backbone.pth")
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

FRAME_STEP  = 1  # Verwerk elk N-de frame (1 = elk frame)

EMOTION_LABELS = [
    "Neutral", "Happiness", "Sadness", "Surprise",
    "Fear", "Disgust", "Anger", "Contempt"
]

# ── CSV-kolommen ──────────────────────────────────────────────────────────────

AU_LABELS = [
    "AU01_intensity",   # Inner Brow Raiser
    "AU02_intensity",   # Outer Brow Raiser
    "AU04_intensity",   # Brow Lowerer
    "AU06_intensity",   # Cheek Raiser
    "AU09_intensity",   # Nose Wrinkler
    "AU12_intensity",   # Lip Corner Puller
    "AU25_intensity",   # Lips Part
    "AU26_intensity",   # Jaw Drop
]

CSV_HEADER = (
    ["frame_index", "timestamp_sec", "face_detected",
     "face_confidence", "face_x1", "face_y1", "face_x2", "face_y2"]
    + [f"emotion_prob_{lbl}" for lbl in EMOTION_LABELS]
    + ["emotion_predicted", "emotion_confidence"]
    + ["gaze_yaw", "gaze_pitch"]
    + AU_LABELS
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def softmax(x: np.ndarray) -> np.ndarray:
    """Berekent de softmax van een 1D array met numerieke stabilisatie.

    Args:
        x: 1D NumPy-array met ruwe logits.

    Returns:
        NumPy-array met kansen die optellen tot 1.
    """
    e = np.exp(x - np.max(x))
    return e / e.sum()


def row_no_face(frame_idx: int, timestamp: float) -> list:
    """Maakt een CSV-rij aan voor een frame zonder gedetecteerd gezicht.

    Alle kolommen na ``face_detected`` worden leeg gelaten zodat downstream
    analyses gemakkelijk frames zonder gezicht kunnen filteren.

    Args:
        frame_idx: Volgnummer van het frame binnen de video.
        timestamp: Tijdstempel in seconden.

    Returns:
        Lijst met waarden overeenkomend met ``CSV_HEADER``.
        ``face_detected`` staat op ``False``; de overige velden zijn leeg.
    """
    return [frame_idx, round(timestamp, 4), False] + [""] * (len(CSV_HEADER) - 3)


_TMP_FRAME_PATH = os.path.join(tempfile.gettempdir(), "_openface_frame.bmp")


def process_video(
    video_path: Path,
    face_detector: FaceDetector,
    multitask_model: MultitaskPredictor,
    output_dir: Path,
) -> None:
    """Analyseert één videobestand en schrijft de resultaten naar een CSV.

    Per frame (of elk N-de frame, zie FRAME_STEP) wordt het volgende gedaan:

    1. Frame als tijdelijk BMP-bestand opslaan.
    2. Gezicht detecteren met FaceDetector; bij geen gezicht een lege rij schrijven.
    3. Bij een gedetecteerd gezicht: emoties, blikrichting en Action Units voorspellen
       met MultitaskPredictor.
    4. Emotielogits omzetten naar softmax-kansen; de klasse met de hoogste kans
       opslaan als voorspelling.
    5. Resultaten als rij wegschrijven naar de CSV.

    Args:
        video_path:      Pad naar het te verwerken videobestand.
        face_detector:   Geladen FaceDetector-instantie (RetinaFace).
        multitask_model: Geladen MultitaskPredictor-instantie (emotie/gaze/AU).
        output_dir:      Map waarin de uitvoer-CSV wordt opgeslagen.
                         De bestandsnaam is gelijk aan die van de video, met .csv extensie.

    Side effects:
        * Maakt _TMP_FRAME_PATH tijdelijk aan en overschrijft dit bij elk frame.
        * Schrijft een CSV-bestand naar output_dir / video_path.stem + ".csv".
        * Drukt voortgang af naar stdout.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [!] Kan video niet openen: {video_path.name}")
        return

    fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    csv_path = output_dir / (video_path.stem + ".csv")

    print(f"  -> {video_path.name}  |  {total} frames  |  {fps:.1f} fps  |  device={DEVICE}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)

        frame_idx = 0
        written   = 0

        while True:
            ret, bgr_frame = cap.read()
            if not ret:
                break

            if frame_idx % FRAME_STEP == 0:
                timestamp = frame_idx / fps

                cv2.imwrite(_TMP_FRAME_PATH, bgr_frame)
                cropped_face, dets = face_detector.get_face(_TMP_FRAME_PATH)

                if cropped_face is None or dets is None:
                    writer.writerow(row_no_face(frame_idx, timestamp))
                else:
                    best = dets[np.argmax(dets[:, 4])]
                    face_conf = round(float(best[4]), 4)
                    x1, y1, x2, y2 = (round(float(best[0])), round(float(best[1])),
                                      round(float(best[2])), round(float(best[3])))

                    emotion_logits, gaze_output, au_output = multitask_model.predict(cropped_face)

                    logits_np  = emotion_logits.cpu().numpy().flatten()
                    probs      = softmax(logits_np)
                    top_idx    = int(np.argmax(probs))
                    top_label  = EMOTION_LABELS[top_idx] if top_idx < len(EMOTION_LABELS) else str(top_idx)
                    confidence = round(float(probs[top_idx]), 4)

                    # Blikrichting
                    gaze = gaze_output.cpu().numpy().flatten() if hasattr(gaze_output, "cpu") else np.array(gaze_output).flatten()
                    gaze_yaw, gaze_pitch = (round(float(gaze[0]), 4), round(float(gaze[1]), 4)) if len(gaze) >= 2 else ("", "")

                    # Action Units
                    au = au_output.cpu().numpy().flatten() if hasattr(au_output, "cpu") else np.array(au_output).flatten()
                    au_values = [round(float(v), 4) for v in au]
                    au_values += [""] * max(0, len(AU_LABELS) - len(au_values))

                    row = (
                        [frame_idx, round(timestamp, 4), True,
                         face_conf, x1, y1, x2, y2]
                        + [round(float(v), 4) for v in probs]
                        + [top_label, confidence]
                        + [gaze_yaw, gaze_pitch]
                        + au_values
                    )
                    writer.writerow(row)
                    written += 1

            frame_idx += 1

    cap.release()
    print(f"     Klaar - {written} frames met gezicht geschreven naar {csv_path.name}")


def main() -> None:
    """Hoofdfunctie: laadt modellen en verwerkt alle video's in VIDEO_DIR.

    Stappen:

    1. Zoek alle videobestanden in VIDEO_DIR op basis van extensie.
    2. Maak OUTPUT_DIR aan indien die nog niet bestaat.
    3. Laad FaceDetector en MultitaskPredictor eenmalig (duur op GPU/CPU).
    4. Roep process_video aan voor elk gevonden videobestand.
    5. Verwijder het tijdelijke BMP-frame na afloop.

    Ondersteunde video-extensies: .mp4, .avi, .mov, .mkv,
    .wmv, .flv, .webm.

    Beëindigt het proces met exit-code 1 als er geen video's worden gevonden.
    """
    video_dir  = Path(VIDEO_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
    videos = sorted([p for p in video_dir.iterdir() if p.suffix.lower() in video_extensions])

    if not videos:
        print(f"Geen video's gevonden in '{video_dir}'.")
        sys.exit(1)

    print(f"Gevonden: {len(videos)} video('s)  |  output -> '{output_dir}'\n")

    face_detector   = FaceDetector(model_path=FACE_MODEL, device=DEVICE)
    multitask_model = MultitaskPredictor(model_path=MULTI_MODEL, device=DEVICE)

    for video_path in videos:
        process_video(video_path, face_detector, multitask_model, output_dir)

    if os.path.exists(_TMP_FRAME_PATH):
        os.remove(_TMP_FRAME_PATH)

    print("\nAlles klaar!")


if __name__ == "__main__":
    main()