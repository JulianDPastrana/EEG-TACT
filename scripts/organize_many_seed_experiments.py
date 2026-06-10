from __future__ import annotations

import shutil
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "many_seed_experiments"
LEGACY_DIR = RESULTS_DIR / "many seeds"


def clean_name(name: str) -> str:
    text = name.strip().lower().replace(" ", "_").replace("-", "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def folder_name_from_root(root_name: str) -> str:
    text = clean_name(root_name)
    if "results_ourmodel_10seeds" in text:
        return "eegformer_10seeds"
    if "results_cnn_lstm_eegnet_10seeds" in text:
        return "cnn_lstm_eegnet_10seeds"
    if "results_tgarnet_10seeds" in text:
        return "tgarnet_10seeds"
    if "results_eegnet_10seeds" in text:
        return "eegnet_10seeds"
    if "results_shallowconvnet_10seeds" in text:
        return "shallowconvnet_10seeds"
    if "results_imcbgt_10seeds" in text:
        return "imcbgt_10seeds"
    if "results_multistream_10_seed" in text or "results_multistream_10seed" in text or "multistream" in text:
        return "multistream_10seeds"
    return text


def archive_root_name(zip_path: Path) -> str | None:
    with zipfile.ZipFile(zip_path) as archive:
        names = [name for name in archive.namelist() if name.endswith("all_seed_summaries.csv") or name.endswith("all_fold_results.csv")]
    if not names:
        return None
    return names[0].split("/", 1)[0]


def extract_archive(zip_path: Path) -> Path | None:
    root_name = archive_root_name(zip_path)
    if root_name is None:
        return None

    target_dir = OUTPUT_DIR / folder_name_from_root(root_name)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            parts = member_path.parts
            if not parts:
                continue
            if parts[0] == root_name:
                relative_parts = parts[1:]
            else:
                relative_parts = parts
            if not relative_parts:
                continue
            destination = target_dir.joinpath(*relative_parts)
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, open(destination, "wb") as output_handle:
                shutil.copyfileobj(source, output_handle)
    return target_dir


def move_legacy_folder() -> Path | None:
    if not LEGACY_DIR.exists():
        return None
    target_dir = OUTPUT_DIR / "eegformer_5fold_run"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.move(str(LEGACY_DIR), str(target_dir))
    return target_dir


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    extracted_rows: list[tuple[str, str]] = []
    for zip_path in sorted(RESULTS_DIR.glob("*.zip")):
        extracted_dir = extract_archive(zip_path)
        if extracted_dir is None:
            continue
        extracted_rows.append((zip_path.name, extracted_dir.name))
        zip_path.unlink()

    legacy_dir = move_legacy_folder()

    print("Organized many-seed experiments")
    for zip_name, folder_name in extracted_rows:
        print(f"{zip_name} -> {folder_name}")
    if legacy_dir is not None:
        print(f"moved legacy folder -> {legacy_dir.name}")
    print(f"\nSaved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()