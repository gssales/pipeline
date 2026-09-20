import re
import json
import csv
from pathlib import Path
from argparse import ArgumentParser

def parse_eval_txt(fps_path: Path):
    """
    Expected:
      psnr: 00.000
      ssim: 00.000
      lpips: 00.000
      psnr_pbr: 00.000
      ssim_pbr: 00.000
      lpips_pbr: 00.000
    Returns (psnr, ssim, lpips, psnr_pbr, ssim_pbr, lpips_pbr) keeping original formatting as strings.
    """
    
    lines = []
    with open(fps_path, "r", encoding="utf-8") as fp:
        lines = [fp.readline().strip() for _ in range(6)]

    psnr_value = ""
    ssim_value = ""
    lpips_value = ""
    psnr_pbr_value = ""
    ssim_pbr_value = ""
    lpips_pbr_value = ""
    for line in lines:
        if line.lower().startswith("psnr:") or line.lower().startswith("psnr_radiance:"):
            psnr_value = line.split(":", 1)[1].strip()
        elif line.lower().startswith("ssim:") or line.lower().startswith("ssim_radiance:"):
            ssim_value = line.split(":", 1)[1].strip()
        elif line.lower().startswith("lpips:") or line.lower().startswith("lpips_radiance:"):
            lpips_value = line.split(":", 1)[1].strip()
        elif line.lower().startswith("psnr_pbr:"):
            psnr_pbr_value = line.split(":", 1)[1].strip()
        elif line.lower().startswith("ssim_pbr:"):
            ssim_pbr_value = line.split(":", 1)[1].strip()
        elif line.lower().startswith("lpips_pbr:"):
            lpips_pbr_value = line.split(":", 1)[1].strip()

    return psnr_value, ssim_value, lpips_value, psnr_pbr_value, ssim_pbr_value, lpips_pbr_value

def parse_fps_txt(fps_path: Path):
    """
    Expected:
      fps: 123.45
      count: 999
    Returns (fps_str, count_str) keeping original formatting as strings.
    """
    fps_value = ""
    count_value = ""
    with open(fps_path, "r", encoding="utf-8") as fp:
        line1 = fp.readline().strip()
        line2 = fp.readline().strip()

    if line1.lower().startswith("fps:"):
        fps_value = line1.split(":", 1)[1].strip()
    else:
        fps_value = line1.strip()

    if line2.lower().startswith("count:"):
        count_value = line2.split(":", 1)[1].strip()
    else:
        count_value = line2.strip()

    return fps_value, count_value


def extract_iteration(key: str):
    """
    Extract a numeric iteration from keys like:
      'ref_gs_30000' -> 30000
      'ours_31000'   -> 31000
    If none found, return -1 so it loses in max().
    """
    m = re.findall(r"(\d+)", key)
    if not m:
        return -1
    return int(m[-1])


def pick_best_key(results: dict):
    """
    Pick the key with the highest numeric iteration.
    Fallback: first key if parsing fails.
    """
    if not results:
        return None

    keys = list(results.keys())
    best = max(keys, key=lambda k: extract_iteration(k))
    return best


def find_scene_dirs(base_path: Path):
    if (base_path / "stage2").exists():
        return [base_path]

    scene_dirs = []
    for item in base_path.iterdir():
        if item.is_dir():
            sub = find_scene_dirs(item)
            if sub is not None:
                scene_dirs.extend(sub)
    return scene_dirs

def parse_usage(base_path: Path, file_name: str = "usage.csv"):
    peak_ram, peak_vram = 0.0, 0.0
    mean_ram, mean_vram = 0.0, 0.0
    count = 0
    if (base_path / file_name).exists():
        with open(base_path / file_name, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) < 3:
                    continue
                try:
                    ram = float(row[1].replace(",", "."))
                    vram = float(row[2].replace(",", "."))
                    peak_ram = max(peak_ram, ram)
                    peak_vram = max(peak_vram, vram)
                    mean_ram += ram
                    mean_vram += vram
                    count += 1
                except ValueError:
                    continue
    if count > 0:
        mean_ram /= count
        mean_vram /= count
    return peak_ram, peak_vram, mean_ram, mean_vram

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
        fps_path = scene_dir / "fps.txt"
        if (scene_dir / "stage2" / "metrics_eval_test.txt").exists():
            eval_path = scene_dir / "stage2" / "metrics_eval_test.txt"
        else:
            eval_path = scene_dir / "stage2" / "eval" / "eval.txt"

        fps_value = ""
        count_value = ""
        if fps_path.exists():
            fps_value, count_value = parse_fps_txt(fps_path)
        else:
            missing.append((fps_path, "fps.txt missing"))

        psnr_value = ""
        ssim_value = ""
        lpips_value = ""
        psnr_pbr_value = ""
        ssim_pbr_value = ""
        lpips_pbr_value = ""
        if eval_path.exists():
            psnr_value, ssim_value, lpips_value, psnr_pbr_value, ssim_pbr_value, lpips_pbr_value = parse_eval_txt(eval_path)
        else:
            missing.append((eval_path, "eval.txt missing"))

        peak_ram1, peak_vram1, mean_ram1, mean_vram1 = parse_usage(scene_dir, file_name="usage_stage1.csv")
        peak_ram2, peak_vram2, mean_ram2, mean_vram2 = parse_usage(scene_dir, file_name="usage_stage2.csv")
        peak_ram_bake, peak_vram_bake, mean_ram_bake, mean_vram_bake = parse_usage(scene_dir, file_name="usage_baking.csv")

        rows.append({
            "scene": str(scene_dir.relative_to(output_root)),
            "key": "Hybrid",
            "PSNR": str(psnr_value).replace(".", ","),
            "SSIM": str(ssim_value).replace(".", ","),
            "LPIPS": str(lpips_value).replace(".", ","),
            "fps": str(fps_value).replace(".", ","),
            "count": count_value,
            "peak_ram1": str(peak_ram1).replace(".", ","),
            "peak_vram1": str(peak_vram1).replace(".", ","),
            "mean_ram1": str(mean_ram1).replace(".", ","),
            "mean_vram1": str(mean_vram1).replace(".", ","),
            "peak_ram2": str(peak_ram2).replace(".", ","),
            "peak_vram2": str(peak_vram2).replace(".", ","),
            "mean_ram2": str(mean_ram2).replace(".", ","),
            "mean_vram2": str(mean_vram2).replace(".", ","),
            "peak_ram_bake": str(peak_ram_bake).replace(".", ","),
            "peak_vram_bake": str(peak_vram_bake).replace(".", ","),
            "mean_ram_bake": str(mean_ram_bake).replace(".", ","),
            "mean_vram_bake": str(mean_vram_bake).replace(".", ","),
        })
        
        rows.append({
            "scene": str(scene_dir.relative_to(output_root)),
            "key": "PBR",
            "PSNR": str(psnr_pbr_value).replace(".", ","),
            "SSIM": str(ssim_pbr_value).replace(".", ","),
            "LPIPS": str(lpips_pbr_value).replace(".", ","),
            "fps": str(fps_value).replace(".", ","),
            "count": count_value,
            "peak_ram1": str(peak_ram1).replace(".", ","),
            "peak_vram1": str(peak_vram1).replace(".", ","),
            "mean_ram1": str(mean_ram1).replace(".", ","),
            "mean_vram1": str(mean_vram1).replace(".", ","),
            "peak_ram2": str(peak_ram2).replace(".", ","),
            "peak_vram2": str(peak_vram2).replace(".", ","),
            "mean_ram2": str(mean_ram2).replace(".", ","),
            "mean_vram2": str(mean_vram2).replace(".", ","),
            "peak_ram_bake": str(peak_ram_bake).replace(".", ","),
            "peak_vram_bake": str(peak_vram_bake).replace(".", ","),
            "mean_ram_bake": str(mean_ram_bake).replace(".", ","),
            "mean_vram_bake": str(mean_vram_bake).replace(".", ","),
        })

    # Write table
    delimiter = "\t" if args.tsv else ","
    fieldnames = ["scene", "key", "PSNR", "SSIM", "LPIPS", "fps", "count", "peak_ram1", "peak_vram1", "mean_ram1", "mean_vram1", "peak_ram2", "peak_vram2", "mean_ram2", "mean_vram2", "peak_ram_bake", "peak_vram_bake", "mean_ram_bake", "mean_vram_bake"]

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

