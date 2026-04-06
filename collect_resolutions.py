from pathlib import Path
from PIL import Image
import csv

ROOT = Path("/mnt/output")
OUTPUT_CSV = "resolutions.csv"

def get_one_image_resolution(render_dir: Path):
    for img_path in render_dir.iterdir():
        if img_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]:
            try:
                with Image.open(img_path) as img:
                    return f"{img.width}x{img.height}"
            except Exception as e:
                print(f"Failed to read {img_path}: {e}")
                return None
    return None

def collect_resolutions(root: Path):
    data = {}  # {method: {scene: resolution}}

    for method_dir in root.iterdir():
        if not method_dir.is_dir():
            continue

        method = method_dir.name
        data.setdefault(method, {})

        for eval_dir in method_dir.glob("eval_*"):
            for dataset_dir in eval_dir.iterdir():
                if not dataset_dir.is_dir():
                    continue

                for scene_dir in dataset_dir.iterdir():
                    render_dir = scene_dir / "test" / "render"
                    if not render_dir.exists():
                        continue

                    scene = scene_dir.name

                    # Only collect once per (method, scene)
                    if scene in data[method]:
                        continue

                    res = get_one_image_resolution(render_dir)
                    if res:
                        data[method][scene] = res

    return data

def write_csv(data, output_path):
    # Collect all scenes across methods
    all_scenes = sorted({scene for m in data.values() for scene in m})

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        
        # Header: method + scenes
        writer.writerow(["method"] + all_scenes)

        for method in sorted(data.keys()):
            row = [method]
            for scene in all_scenes:
                row.append(data[method].get(scene, ""))
            writer.writerow(row)

if __name__ == "__main__":
    data = collect_resolutions(ROOT)
    write_csv(data, OUTPUT_CSV)
    print(f"Saved to {OUTPUT_CSV}")