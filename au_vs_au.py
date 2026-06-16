"""
au_vs_au.py
============
Combineert AU-occurrence-data van beide tools tot één groot CSV-bestand.
Nodig voor analyse 4.2: AU-correlaties PyAFAR vs OpenFace
Per video worden de AU-kolommen van beide tools samengevoegd op framenummer.
Elke rij is één frame. Alle video's worden onder elkaar gezet.

INVOER:
  77 mens def fragmenten voor PyAFAR/
    <video>.csv  →  frame: kolom 'Frame'
                    AU-occurrence: Occ_au_1, Occ_au_2, Occ_au_4,
                                   Occ_au_6, Occ_au_12  (floats 0–1)

  77 mens def fragmenten voor Openface/
    <video>.csv  →  frame: index 'frame_index'
                    AU-occurrence: AU01_intensity, AU02_intensity,
                                   AU04_intensity, AU06_intensity,
                                   AU12_intensity  (floats 0–1)

UITVOER:
  auPyAFAR_vs_auOpenFace.csv
    video | frame | PyAFAR_AU1 | PyAFAR_AU2 | PyAFAR_AU4 | PyAFAR_AU6 |
    PyAFAR_AU12 | OpenFace_AU1 | OpenFace_AU2 | OpenFace_AU4 |
    OpenFace_AU6 | OpenFace_AU12
"""

from pathlib import Path
import sys
import pandas as pd


# ── Configuratie ──────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PYAFAR_DIR   = SCRIPT_DIR / "data_PyAFAR"
OPENFACE_DIR = SCRIPT_DIR / "data_OpenFace"
OUTPUT_PATH  = SCRIPT_DIR / "auPyAFAR_vs_auOpenFace.csv"

TARGET_AUS = [1, 2, 4, 6, 12]

# PyAFAR-map: Occ_au_* kolommen
PYAFAR_AU_COLS   = {au: f"Occ_au_{au}"          for au in TARGET_AUS}

# OpenFace-map: AU*_intensity kolommen
OPENFACE_AU_COLS = {au: f"AU{au:02d}_intensity"  for au in TARGET_AUS}
# ─────────────────────────────────────────────────────────────────────────────


def load_pyafar(csv_path: Path) -> pd.DataFrame:
    """
    Laad een PyAFAR-CSV (Occ_au_* kolommen, frame in kolom 'Frame').
    """
    df = pd.read_csv(csv_path)

    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    result = pd.DataFrame()
    result["frame"] = df["Frame"].astype(int)

    for au, col in PYAFAR_AU_COLS.items():
        if col in df.columns:
            result[f"PyAFAR_AU{au}"] = df[col].values
        else:
            result[f"PyAFAR_AU{au}"] = float("nan")
            print(f"    [WAARSCHUWING] PyAFAR: kolom '{col}' ontbreekt in {csv_path.name}")

    return result.reset_index(drop=True)


def load_openface(csv_path: Path) -> pd.DataFrame:
    """
    Laad een OpenFace-CSV (AU*_intensity kolommen, frame als index 'frame_index').
    """
    df = pd.read_csv(csv_path, index_col=0)

    result = pd.DataFrame()
    result["frame"] = df.index.astype(int)

    for au, col in OPENFACE_AU_COLS.items():
        if col in df.columns:
            result[f"OpenFace_AU{au}"] = df[col].values
        else:
            result[f"OpenFace_AU{au}"] = float("nan")
            print(f"    [WAARSCHUWING] OpenFace: kolom '{col}' ontbreekt in {csv_path.name}")

    return result.reset_index(drop=True)


def main() -> None:
    for folder in (PYAFAR_DIR, OPENFACE_DIR):
        if not folder.exists():
            print(f"Map niet gevonden: {folder}")
            sys.exit(1)

    # Bestanden matchen op identieke bestandsnaam in beide mappen
    pyafar_files   = {p.name: p for p in sorted(PYAFAR_DIR.glob("*.csv"))}
    openface_files = {p.name: p for p in sorted(OPENFACE_DIR.glob("*.csv"))}

    all_names = sorted(set(pyafar_files) | set(openface_files))

    if not all_names:
        print("Geen CSV-bestanden gevonden.")
        sys.exit(1)

    print(f"PyAFAR-map   : {PYAFAR_DIR}")
    print(f"OpenFace-map : {OPENFACE_DIR}")
    print(f"Uitvoer      : {OUTPUT_PATH}")
    print(f"AUs          : {TARGET_AUS}")
    print(f"Bestanden    : {len(all_names)}")
    print()

    all_rows = []

    for name in all_names:
        pyafar_path   = pyafar_files.get(name)
        openface_path = openface_files.get(name)

        if pyafar_path is None:
            print(f"  [SKIP] {name} — ontbreekt in PyAFAR-map")
            continue
        if openface_path is None:
            print(f"  [SKIP] {name} — ontbreekt in OpenFace-map")
            continue

        video_name = Path(name).stem
        print(f"  {video_name}")

        pyafar_df   = load_pyafar(pyafar_path)
        openface_df = load_openface(openface_path)

        # Merge op framenummer: frame 0 van PyAFAR koppelt aan frame 0 van
        # OpenFace, etc. Outer join zodat ontbrekende frames zichtbaar blijven.
        merged = pd.merge(pyafar_df, openface_df, on="frame", how="outer")
        merged.sort_values("frame", inplace=True)
        merged.reset_index(drop=True, inplace=True)

        merged.insert(0, "video", video_name)
        all_rows.append(merged)

    print()

    if not all_rows:
        print("Geen video's verwerkt.")
        sys.exit(1)

    combined = pd.concat(all_rows, ignore_index=True)

    au_cols = (
        [f"PyAFAR_AU{au}"   for au in TARGET_AUS] +
        [f"OpenFace_AU{au}" for au in TARGET_AUS]
    )
    combined = combined[["video", "frame"] + au_cols]

    combined.to_csv(OUTPUT_PATH, index=False)

    print(f"Klaar! {len(all_rows)} video('s), {len(combined)} frames totaal.")
    print(f"Opgeslagen in: {OUTPUT_PATH}")
    print()
    print(combined.head(5).to_string(index=False))


if __name__ == "__main__":
    main()