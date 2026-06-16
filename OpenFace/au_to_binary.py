"""
au_to_binary.py
===============
Zet AU-intensiteiten uit OpenFace CSV's om naar 0/1
op basis van één of meer drempelwaardes.

Niet-AU-kolommen (Frame, Pitch, landmarks, …) worden
ongewijzigd meegekopieerd naar de uitvoer.
"""

from pathlib import Path
import re
import sys
import pandas as pd

# ── Configuratie ─────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
INPUT_DIR  = PROJECT_DIR / "data_OpenFace"
OUTPUT_DIR = SCRIPT_DIR / "output_binary_au"    # resultatenmap

THRESHOLDS = [0.3, 0.5, 0.7]
# ─────────────────────────────────────────────────────────────────────────────


def get_au_columns(df: pd.DataFrame) -> list[str]:
    """
    Geef alleen de OpenFace AU-intensiteitskolommen terug,
    zoals AU01_intensity, AU06_intensity en AU12_intensity.
    """
    pattern = re.compile(r"^AU\d+_intensity$", re.IGNORECASE)
    return [c for c in df.columns if pattern.match(c)]


def binarize(df: pd.DataFrame, au_cols: list[str], threshold: float) -> pd.DataFrame:
    """
    Zet AU-waarden om naar 0/1 op basis van drempelwaarde.
    Alle niet-AU-kolommen worden ongewijzigd overgenomen.
    """
    result = df.copy()
    result[au_cols] = (df[au_cols] >= threshold).astype(int)
    return result


def main():
    csv_files = sorted(INPUT_DIR.glob("*.csv"))
    if not csv_files:
        print(f"Geen CSV-bestanden gevonden in '{INPUT_DIR}'.")
        sys.exit(1)

    print(f"Gevonden   : {len(csv_files)} CSV-bestand(en)")
    print(f"Drempels   : {THRESHOLDS}\n")

    # Maak outputmappen aan per drempelwaarde
    threshold_dirs: dict[float, Path] = {}
    for t in THRESHOLDS:
        folder_name = f"threshold_{str(t).replace('.', '_')}"
        path = OUTPUT_DIR / folder_name
        path.mkdir(parents=True, exist_ok=True)
        threshold_dirs[t] = path

    for csv_path in csv_files:
        print(f"  {csv_path.name}")
        df = pd.read_csv(csv_path, index_col=0)

        au_cols = get_au_columns(df)
        if not au_cols:
            print("    [SKIP] Geen AU-kolommen gevonden.")
            continue

        print(f"    AU-kolommen : {au_cols}")

        for threshold, out_dir in threshold_dirs.items():
            binary_df = binarize(df, au_cols, threshold)
            out_path = out_dir / csv_path.name
            binary_df.to_csv(out_path, index=True)

        print(f"    Opgeslagen in {len(THRESHOLDS)} drempelwaarde-mappen")

    print(f"\nKlaar! Resultaten in: {OUTPUT_DIR}")
    for t, path in threshold_dirs.items():
        n = len(list(path.glob("*.csv")))
        print(f"  {path.name}/  ({n} bestand(en))")


if __name__ == "__main__":
    main()