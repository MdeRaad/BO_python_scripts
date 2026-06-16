"""
AU Aggregator Script
====================

Combineert menselijke referentielabels met AU-aanwezigheidsdata
van PyAFAR en OpenFace.

Verwachte mapstructuur:
    <PROJECT_ROOT>/
        referentielabels.csv
        au_summary.csv

        PyAFAR/
            output_binary_au/
                threshold_0_3/  -> <video>.csv
                threshold_0_5/  -> <video>.csv
                threshold_0_7/  -> <video>.csv

        OpenFace/
            output_binary_au/
                threshold_0_3/  -> <video>.csv
                threshold_0_5/  -> <video>.csv
                threshold_0_7/  -> <video>.csv

"""

from pathlib import Path
import sys
import pandas as pd


# ── Configuratie ─────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent

LABELS_FILE = SCRIPT_DIR / "referentielabels.csv"
OUTPUT_FILE = SCRIPT_DIR / "au_summary.csv"

PYAFAR_BINARY_DIR = SCRIPT_DIR / "PyAFAR" / "output_binary_au"
OPENFACE_BINARY_DIR = SCRIPT_DIR / "OpenFace" / "output_binary_au"

AUS = [4, 6, 12]
THRESHOLDS = ["0_3", "0_5", "0_7"]

PYAFAR_AU_COL = {au: f"Occ_au_{au}" for au in AUS}
OPENFACE_AU_COL = {au: f"AU{au:02d}_intensity" for au in AUS}
# ─────────────────────────────────────────────────────────────────────────────


def read_csv_flexible(path: Path) -> pd.DataFrame:
    """
    Lees een CSV-bestand flexibel in.
    Probeert komma, puntkomma en tab als scheidingsteken.
    """
    for sep in [",", ";", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep)

            if df.shape[1] > 1:
                return df

        except Exception:
            continue

    raise ValueError(f"Kan '{path}' niet lezen als CSV.")


def load_labels(path: Path) -> pd.DataFrame:
    """
    Laad referentielabels uit referentielabels.csv.

    Verwacht minimaal deze kolommen:
      video
      menselijk_label
    """
    df = read_csv_flexible(path)
    df.columns = [str(col).strip() for col in df.columns]

    required_cols = ["video", "menselijk_label"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"[FOUT] referentielabels.csv mist kolom(men): {missing_cols}")
        print("Verwachte kolommen: video, menselijk_label")
        sys.exit(1)

    result = df[required_cols].copy()

    result["video"] = result["video"].astype(str).str.strip()
    result["menselijk_label"] = result["menselijk_label"].astype(str).str.strip()

    result = result[
        result["video"].notna()
        & (result["video"] != "nan")
        & (result["video"].str.len() > 0)
    ].reset_index(drop=True)

    return result


def video_stem(video_name: str) -> str:
    """
    Haal de bestandsnaam zonder extensie uit de videonaam.

    Voorbeeld:
      kind_01.mp4 -> kind_01
      kind_01     -> kind_01
    """
    return Path(str(video_name)).stem


def count_au(csv_path: Path, au_col: str) -> tuple[int | None, int | None]:
    """
    Tel hoe vaak een AU-kolom actief is in een gebinariseerde CSV.

    Geeft terug:
      aanwezig = aantal frames waarin AU actief is
      totaal   = totaal aantal frames
    """
    if not csv_path.exists():
        print(f"  [NIET GEVONDEN] {csv_path}")
        return None, None

    try:
        df = pd.read_csv(csv_path)

    except Exception as error:
        print(f"  [FOUT] Lezen {csv_path}: {error}")
        return None, None

    # Verwijder eventuele index-kolom die door eerdere scripts is meegeschreven.
    unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    if au_col not in df.columns:
        print(f"  [WAARSCHUWING] Kolom '{au_col}' niet gevonden in {csv_path.name}")
        return None, None

    aanwezig = int(df[au_col].fillna(0).sum())
    totaal = len(df)

    return aanwezig, totaal


def main() -> None:
    if not LABELS_FILE.exists():
        print(f"[FOUT] Labelbestand niet gevonden: {LABELS_FILE}")
        print("Maak eerst referentielabels.csv aan in de hoofdmap van het project.")
        sys.exit(1)

    labels = load_labels(LABELS_FILE)

    print(f"Referentielabels: {LABELS_FILE}")
    print(f"PyAFAR-map      : {PYAFAR_BINARY_DIR}")
    print(f"OpenFace-map    : {OPENFACE_BINARY_DIR}")
    print(f"Outputbestand   : {OUTPUT_FILE}")
    print(f"AUs             : {AUS}")
    print(f"Drempelwaardes  : {THRESHOLDS}")
    print()
    print(f"Geladen: {len(labels)} video's uit referentielabels.csv")
    print()

    rows = []

    for _, row in labels.iterrows():
        video_name = str(row["video"]).strip()
        video_label = str(row["menselijk_label"]).strip()
        stem = video_stem(video_name)

        record = {
            "video": video_name,
            "menselijk_label": video_label,
        }

        pyafar_totaal = None

        for au in AUS:
            col = PYAFAR_AU_COL[au]

            for thresh in THRESHOLDS:
                csv_path = PYAFAR_BINARY_DIR / f"threshold_{thresh}" / f"{stem}.csv"

                aanwezig, totaal = count_au(csv_path, col)

                record[f"pyafar_AU{au}_t{thresh}_aanwezig"] = aanwezig

                if totaal is not None and pyafar_totaal is None:
                    pyafar_totaal = totaal

                proportie = (
                    round(aanwezig / totaal, 4)
                    if aanwezig is not None and totaal and totaal > 0
                    else None
                )

                record[f"pyafar_AU{au}_t{thresh}_proportie"] = proportie

        record["pyafar_totaal_frames"] = pyafar_totaal

        openface_totaal = None

        for au in AUS:
            col = OPENFACE_AU_COL[au]

            for thresh in THRESHOLDS:
                csv_path = OPENFACE_BINARY_DIR / f"threshold_{thresh}" / f"{stem}.csv"

                aanwezig, totaal = count_au(csv_path, col)

                record[f"openface_AU{au}_t{thresh}_aanwezig"] = aanwezig

                if totaal is not None and openface_totaal is None:
                    openface_totaal = totaal

                proportie = (
                    round(aanwezig / totaal, 4)
                    if aanwezig is not None and totaal and totaal > 0
                    else None
                )

                record[f"openface_AU{au}_t{thresh}_proportie"] = proportie

        record["openface_totaal_frames"] = openface_totaal

        rows.append(record)
        print(f"  Verwerkt: {video_name}")

    df_out = pd.DataFrame(rows)

    ordered_cols = ["video", "menselijk_label"]

    # PyAFAR: aanwezig per AU per threshold, dan totaal frames, dan proporties
    for au in AUS:
        for thresh in THRESHOLDS:
            ordered_cols.append(f"pyafar_AU{au}_t{thresh}_aanwezig")

    ordered_cols.append("pyafar_totaal_frames")

    for au in AUS:
        for thresh in THRESHOLDS:
            ordered_cols.append(f"pyafar_AU{au}_t{thresh}_proportie")

    # OpenFace: aanwezig per AU per threshold, dan totaal frames, dan proporties
    for au in AUS:
        for thresh in THRESHOLDS:
            ordered_cols.append(f"openface_AU{au}_t{thresh}_aanwezig")

    ordered_cols.append("openface_totaal_frames")

    for au in AUS:
        for thresh in THRESHOLDS:
            ordered_cols.append(f"openface_AU{au}_t{thresh}_proportie")

    ordered_cols = [col for col in ordered_cols if col in df_out.columns]
    df_out = df_out[ordered_cols]

    df_out.to_csv(
        OUTPUT_FILE,
        index=False,
        sep=",",
        decimal=".",
        encoding="utf-8-sig",
        float_format="%.4f"
    )

    print(f"\nKlaar! Opgeslagen in: {OUTPUT_FILE}")
    print(f"Rijen: {len(df_out)} | Kolommen: {len(df_out.columns)}")
    print("\nKolomnamen:")

    for col in df_out.columns:
        print(f"  {col}")


if __name__ == "__main__":
    main()