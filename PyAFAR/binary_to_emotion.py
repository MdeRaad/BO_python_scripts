"""
binary_to_emotion.py
===================

Zet gebinariseerde PyAFAR AU-output om naar een AU-gebaseerd
fragmentlabel voor alleen happiness en sadness.

Dit script verwacht al gebinariseerde AU-data. 
De omzetting van AU-kansen naar 0/1 gebeurt dus niet hier, maar eerder met au_to_binary.py.

INVOER:
  output_binary_au/
    threshold_0_3/
    threshold_0_5/
    threshold_0_7/

UITVOER:
  output_emotion_labels/
    per_video/
      threshold_0_3/  <video>.csv
      threshold_0_5/  <video>.csv
      threshold_0_7/  <video>.csv

    video_summary.csv
      Compact overzicht:
      video | au_based_label_t0.3 | happiness_score_t0.3 | sadness_score_t0.3 | ...

    video_details.csv
      Detailoverzicht per video en threshold.
      Bevat het AU-gebaseerde label, de happiness- en sadness-score,
      de gebruikte windowgrootte en per gebruikte AU de proportie frames
      waarin deze actief was.

METHODE:
    Per freagment wordt een window-methode gebruikt. Daarbij hoeven twee AUs 
    niet exact in hetzelfde frame actief te zijn, maar wel binnen dezelfde 
    korte tijdsperiode (window).

    Happiness-window:
    AU6 en AU12 komen allebei minimaal één keer voor binnen hetzelfde window.

    Sadness-window:
    AU4 komt minimaal één keer voor binnen hetzelfde window.

    Per fragment wordt berekend welk percentage van de windows voldoet aan
    het happiness-patroon en welk percentage voldoet aan het sadness-patroon.

    Daarna wordt per fragment een AU-gebaseerd label toegekend:

    happiness:
        happiness-score is groter dan sadness-score.

    sadness:
        sadness-score is groter dan happiness-score.

    ambiguous:
        happiness-score en sadness-score zijn gelijk, maar beide groter dan 0.

    undetermined:
        beide scores zijn 0.

"""

from pathlib import Path
import re
import sys
import pandas as pd


# ── Configuratie ─────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent

# Deze map moet de output zijn van au_to_binary.py
BINARY_DIR = SCRIPT_DIR / "output_binary_au"

# Aparte outputmap voor PyAFAR-labels
OUTPUT_DIR = SCRIPT_DIR / "output_emotion_labels"

TARGET_AUS = [4, 6, 12]

WINDOW_SIZE_FRAMES = 10
# ─────────────────────────────────────────────────────────────────────────────


def get_au_column_map(df: pd.DataFrame) -> dict[int, str]:
    """
    Bouw {au_nummer: kolomnaam} uitsluitend op basis van Occ_au_* kolommen.

    Verwacht kolommen zoals:
      Occ_au_4
      Occ_au_6
      Occ_au_12
    """
    pattern = re.compile(r"^occ_au_(\d+)$", re.IGNORECASE)
    result = {}

    for col in df.columns:
        match = pattern.match(col)

        if match:
            au_number = int(match.group(1))

            if au_number in TARGET_AUS:
                result[au_number] = col

    return result


def validate_binary(df: pd.DataFrame, au_map: dict[int, str]) -> None:
    """
    Gooi een fout als een AU-kolom niet-binaire waarden bevat.
    Dit script verwacht dus alleen 0/1-waarden in de AU-kolommen.
    """
    for col in set(au_map.values()):
        unique_values = set(df[col].dropna().unique())

        if not unique_values.issubset({0, 1}):
            raise ValueError(
                f"Kolom '{col}' bevat niet-binaire waarden: {unique_values}. "
                "Verwerk de data eerst met au_to_binary.py."
            )


def get_au_series(df: pd.DataFrame, au_map: dict[int, str], au: int) -> pd.Series:
    """
    Haal een AU-kolom op als 0/1-serie.
    Als de AU-kolom ontbreekt, wordt een serie met alleen nullen teruggegeven.
    """
    col = au_map.get(au)

    if col is None:
        return pd.Series(0, index=df.index)

    return df[col].fillna(0).astype(int)


def calculate_window_score(au_a: pd.Series, au_b: pd.Series, window_size: int) -> float:
    """
    Berekent de proportie windows waarin beide AUs minimaal één keer voorkomen.

    Voorbeeld:
      Twee AUs hoeven niet exact in hetzelfde frame actief te zijn,
      maar moeten wel allebei binnen hetzelfde window voorkomen.

    Als au_a en au_b dezelfde serie zijn (één-AU-modus), wordt de score de
    proportie windows waarin die ene AU minimaal één keer actief is.
    """
    n_frames = len(au_a)

    if n_frames == 0:
        return 0.0

    # Als een fragment korter is dan het window, gebruiken we het hele fragment als één window.
    if n_frames < window_size:
        a_present = au_a.sum() > 0
        b_present = au_b.sum() > 0
        return 1.0 if a_present and b_present else 0.0

    hits = 0
    total_windows = 0

    for start in range(0, n_frames - window_size + 1):
        end = start + window_size

        a_present = au_a.iloc[start:end].sum() > 0
        b_present = au_b.iloc[start:end].sum() > 0

        if a_present and b_present:
            hits += 1

        total_windows += 1

    return hits / total_windows if total_windows > 0 else 0.0


def calculate_fragment_scores(df: pd.DataFrame, au_map: dict[int, str]) -> dict[str, object]:
    """
    Bereken per fragment:
      - proportie actieve frames per losse AU
      - happiness-score op basis van AU6 en AU12 binnen hetzelfde window
      - sadness-score op basis van AU4 alleen
    """
    au4 = get_au_series(df, au_map, 4)
    au6 = get_au_series(df, au_map, 6)
    au12 = get_au_series(df, au_map, 12)

    happiness_score = calculate_window_score(au6, au12, WINDOW_SIZE_FRAMES)

    # AU4 alleen: au4 wordt twee keer meegegeven zodat een window scoort
    # zodra AU4 minimaal één keer actief is binnen dat window.
    sadness_score = calculate_window_score(au4, au4, WINDOW_SIZE_FRAMES)

    scores = {
        "window_size_frames": WINDOW_SIZE_FRAMES,

        "AU4_frame_proportion": round(float(au4.mean()), 4),
        "AU6_frame_proportion": round(float(au6.mean()), 4),
        "AU12_frame_proportion": round(float(au12.mean()), 4),

        "happiness_score": round(float(happiness_score), 4),
        "sadness_score": round(float(sadness_score), 4),
    }

    return scores


def assign_happiness_sadness_label(scores: dict[str, object]) -> str:
    """
    Zet window-scores om naar één AU-gebaseerd label.
    """
    happiness_score = scores["happiness_score"]
    sadness_score = scores["sadness_score"]

    if happiness_score == 0 and sadness_score == 0:
        return "undetermined"

    if happiness_score > sadness_score:
        return "happiness"

    if sadness_score > happiness_score:
        return "sadness"

    return "ambiguous"


def add_fragment_summary_to_dataframe(
    df: pd.DataFrame,
    scores: dict[str, object],
    label: str
) -> pd.DataFrame:
    """
    Voeg fragmentinformatie toe aan de per-video CSV.
    Omdat het label op fragmentniveau wordt bepaald, krijgt elke rij/frame
    dezelfde fragment-level samenvatting.
    """
    result_df = df.copy()

    result_df["au_based_label"] = label

    for key, value in scores.items():
        result_df[key] = value

    return result_df


def threshold_to_label(threshold_name: str) -> str:
    """
    Maakt van threshold_0_3 bijvoorbeeld t0.3.
    """
    return threshold_name.replace("threshold_", "t").replace("_", ".")


def process_file(csv_path: Path, out_dir: Path, threshold_name: str) -> dict[str, object] | None:
    """
    Verwerk één gebinariseerde CSV.
    Slaat een per-video CSV op en geeft een rij terug voor video_details.csv.
    """
    df = pd.read_csv(csv_path)

    # Verwijder eventuele index-kolom die door eerdere scripts is meegeschreven.
    unnamed_cols = [col for col in df.columns if str(col).startswith("Unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    au_map = get_au_column_map(df)

    if not au_map:
        print(f"    [SKIP] Geen AU-kolommen in {csv_path.name}")
        return None

    try:
        validate_binary(df, au_map)
    except ValueError as error:
        print(f"    [FOUT] {csv_path.name}: {error}")
        return None

    scores = calculate_fragment_scores(df, au_map)
    label = assign_happiness_sadness_label(scores)

    result_df = add_fragment_summary_to_dataframe(df, scores, label)
    result_df.to_csv(out_dir / csv_path.name, index=False)

    row = {
        "video": csv_path.stem,
        "threshold": threshold_to_label(threshold_name),
        "au_based_label": label,
    }

    row.update(scores)

    return row


def main() -> None:
    threshold_dirs = sorted(BINARY_DIR.glob("threshold_*"))

    if not threshold_dirs:
        print(f"Geen threshold-mappen gevonden in '{BINARY_DIR}'.")
        print("Draai eerst au_to_binary.py.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Binary map    : {BINARY_DIR}")
    print(f"Uitvoermap    : {OUTPUT_DIR}")
    print(f"Drempelwaardes: {[directory.name for directory in threshold_dirs]}")
    print(f"Windowgrootte : {WINDOW_SIZE_FRAMES} frames")
    print(f"AU-regels     : happiness = AU6 + AU12 binnen hetzelfde window")
    print(f"                sadness   = AU4 alleen")
    print()

    results: dict[str, dict[str, dict[str, object]]] = {}
    detail_rows = []

    for threshold_dir in threshold_dirs:
        csv_files = sorted(threshold_dir.glob("*.csv"))

        if not csv_files:
            print(f"  [LEEG] {threshold_dir.name} — geen CSV's, overgeslagen.")
            continue

        per_video_out = OUTPUT_DIR / "per_video" / threshold_dir.name
        per_video_out.mkdir(parents=True, exist_ok=True)

        print(f"  {threshold_dir.name}/  ({len(csv_files)} bestand(en))")

        for csv_path in csv_files:
            video_result = process_file(csv_path, per_video_out, threshold_dir.name)

            if video_result is None:
                continue

            video_stem = csv_path.stem
            results.setdefault(video_stem, {})[threshold_dir.name] = video_result
            detail_rows.append(video_result)

        print()

    if not results:
        print("Geen bestanden verwerkt.")
        return

    threshold_names = [directory.name for directory in threshold_dirs]

    # CSV 1: compact overzicht met AU-gebaseerde labels per threshold
    summary_rows = []

    for video_stem, threshold_results in sorted(results.items()):
        row = {"video": video_stem}

        for threshold_name in threshold_names:
            label_col = f"au_based_label_{threshold_to_label(threshold_name)}"
            happiness_col = f"happiness_score_{threshold_to_label(threshold_name)}"
            sadness_col = f"sadness_score_{threshold_to_label(threshold_name)}"

            threshold_result = threshold_results.get(threshold_name, {})

            row[label_col] = threshold_result.get("au_based_label", "")
            row[happiness_col] = threshold_result.get("happiness_score", "")
            row[sadness_col] = threshold_result.get("sadness_score", "")

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # CSV 2: detailoverzicht met gebruikte scores en frame-proporties
    details_df = pd.DataFrame(detail_rows)

    detail_order = [
        "video",
        "threshold",
        "au_based_label",
        "window_size_frames",
        "happiness_score",
        "sadness_score",
        "AU4_frame_proportion",
        "AU6_frame_proportion",
        "AU12_frame_proportion",
    ]

    existing_detail_order = [col for col in detail_order if col in details_df.columns]
    remaining_detail_cols = [
        col for col in details_df.columns
        if col not in existing_detail_order
    ]
    details_df = details_df[existing_detail_order + remaining_detail_cols]

    summary_path = OUTPUT_DIR / "video_summary.csv"
    details_path = OUTPUT_DIR / "video_details.csv"

    summary_df.to_csv(summary_path, index=False)
    details_df.to_csv(details_path, index=False)

    print("─" * 50)
    print(f"Klaar! {len(summary_rows)} video('s) verwerkt.")
    print(f"  Per-video CSV's → {OUTPUT_DIR / 'per_video'}/")
    print(f"  Summary CSV     → {summary_path}")
    print(f"  Details CSV     → {details_path}")
    print()
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()