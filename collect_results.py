import json
import csv
from pathlib import Path
from argparse import ArgumentParser

def find_scene_dirs(base_path: Path):
    if (base_path / "cfg_args").exists():
        return [base_path]

    scene_dirs = []
    for item in base_path.iterdir():
        if item.is_dir():
            sub = find_scene_dirs(item)
            if sub is not None:
                scene_dirs.extend(sub)
    return scene_dirs

def main():
    parser = ArgumentParser(description="Collect PSNR/SSIM/LPIPS + FPS into a CSV across all scenes.")
    parser.add_argument("--output_path", default=r"E:\\output\\ours\\eval",
                        help="Root eval output path (contains camera_regular/...)")
    parser.add_argument("--csv_name", default="results_all.csv",
                        help="Output CSV filename (written inside output_path)")
    parser.add_argument("--tsv", action="store_true",
                        help="Write TSV instead of CSV")
    args, _ = parser.parse_known_args()

    output_root = Path(args.output_path)
    out_file = output_root / args.csv_name

    scene_dirs = find_scene_dirs(output_root)
    print(f"Found {len(scene_dirs)} scene folders under {output_root}")

    rows = []
    missing = []

    for scene_dir in scene_dirs:
        results_path = scene_dir / "collected_results.json"

        if not results_path.exists():
            missing.append((scene_dir, "collected_results.json missing"))
            continue

        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)

            rows.append({
                "scene": str(scene_dir.relative_to(output_root)),
                "PSNR": str(results.get("PSNR", "")).replace(".", ","),
                "SSIM": str(results.get("SSIM", "")).replace(".", ","),
                "LPIPS": str(results.get("LPIPS", "")).replace(".", ","),
                "MAE": str(results.get("MAE", "")).replace(".", ","),
                "fps": str(results.get("FPS", "")).replace(".", ","),
                "count": results.get("count", ""),
                "peak_ram": str(results.get("peak_ram", "")).replace(".", ","),
                "peak_vram": str(results.get("peak_vram", "")).replace(".", ","),
                "training_time": str(results.get("training_time", "")).replace(".", ","),
            })

    # Write table
    delimiter = "\t" if args.tsv else ","
    fieldnames = ["scene", "PSNR", "SSIM", "LPIPS", "MAE", "fps", "count", "peak_ram", "peak_vram", "training_time"]

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} rows to: {out_file}")

    if missing:
        print("\nWarnings (missing files):")
        for scene_dir, reason in missing:
            print(f"  - {scene_dir}: {reason}")


if __name__ == "__main__":
    main()

