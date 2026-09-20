import re
import json
import csv
from pathlib import Path
from argparse import ArgumentParser

from utils.parser_utils import load_scene_args

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

def parse_results_json(results_path: Path):
  if not results_path.exists():
    # raise FileNotFoundError(f"Results file not found: {results_path}")
    print(f"Warning: Results file not found: {results_path}. Returning empty results.")
    return {}
  with open(results_path, "r", encoding="utf-8") as f:
    results = json.load(f)
    has_method_key = True
    for key in results.keys():
      if "psnr" in key.lower() or "ssim" in key.lower() or "lpips" in key.lower():
        has_method_key = False
        break
    if has_method_key:
      best_key = pick_best_key(results)
      if best_key is None or best_key not in results:
        raise ValueError(f"Results file is empty or invalid: {results_path}")
      return results[best_key]
    else:
      parsed_results = {}
      for key in results.keys():
        if "psnr" in key.lower():
          parsed_results["PSNR"] = results[key]
        elif "ssim" in key.lower():
          parsed_results["SSIM"] = results[key]
        elif "lpips" in key.lower():
          parsed_results["LPIPS"] = results[key]
      return parsed_results


def parse_fps_txt(fps_path: Path):
  """
  Expected:
    fps: 123.45
    count: 999
  Returns (fps_str, count_str) keeping original formatting as strings.
  """
  if not fps_path.exists():
    # raise FileNotFoundError(f"FPS file not found: {fps_path}")
    print(f"Warning: FPS file not found: {fps_path}. Returning empty values.")
    return "", ""
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

def parse_usage(usage_path: Path):
  if not usage_path.exists():
    # raise FileNotFoundError(f"Usage file not found: {usage_path}")
    print(f"Warning: Usage file not found: {usage_path}. Returning zeros.")
    return 0.0, 0.0, 0.0, 0.0

  peak_ram, peak_vram = 0.0, 0.0
  mean_ram, mean_vram = 0.0, 0.0
  count = 0
  with open(usage_path, "r") as f:
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

def read_latest_training_time(model_path: Path) -> str:
  timestamped_files = sorted(
    model_path.glob("training_time_*.txt")
  )

  if timestamped_files:
    training_time_path = timestamped_files[-1]
  else:
    training_time_path = model_path / "training_time.txt"

  if not training_time_path.exists():
    # raise FileNotFoundError(f"No training time file found in {model_path}")
    print(f"Warning: No training time file found in {model_path}. Returning empty string.")
    return ""

  with training_time_path.open("r", encoding="utf-8") as file:
    return file.readline().strip()
  
def read_latest_stage_time(
  model_path: Path,
  stage: str,
) -> float:
  time_files = sorted(
    model_path.glob(f"training_time_{stage}_*.txt")
  )

  if not time_files:
    # raise FileNotFoundError(
    #   f"No training time file found for stage "
    #   f"'{stage}' in {model_path}"
    # )
    print(f"Warning: No training time file found for stage '{stage}' in {model_path}. Returning 0.0.")
    return 0.0

  latest_file = time_files[-1]

  with latest_file.open("r", encoding="utf-8") as file:
    return float(file.readline().strip())

def get_training_stages(parameters: dict) -> list[str]:
  training_config = parameters["stages"]["training"]
  if "script" in training_config:
    return ["training"]
  return list(training_config.keys())

def collect_multi_stage_training_times(
  model_path: Path,
  parameters: dict,
) -> dict:
  stage_names = get_training_stages(parameters)

  stage_times = {
    stage: read_latest_stage_time(model_path, stage)
    for stage in stage_names
  }

  return {
    "stages": stage_times,
    "total": sum(stage_times.values()),
  }

def collect_multi_stage_usage(model_path: Path, parameters: dict) -> dict:
  stage_names = get_training_stages(parameters)
  stage_usages = {}
  max_peak_ram, max_peak_vram = 0.0, 0.0
  for stage in stage_names:
    usage_path = model_path / f"usage_{stage}.csv"
    peak_ram, peak_vram, mean_ram, mean_vram = parse_usage(usage_path)
    stage_usages[stage] = {
      "peak_ram": peak_ram,
      "peak_vram": peak_vram,
      "mean_ram": mean_ram,
      "mean_vram": mean_vram,
    }
    max_peak_ram = max(max_peak_ram, peak_ram)
    max_peak_vram = max(max_peak_vram, peak_vram)
  return {
    "stages": stage_usages,
    "max_peak_ram": max_peak_ram,
    "max_peak_vram": max_peak_vram,
  }

def main():
  parser = ArgumentParser(description="Collect PSNR/SSIM/LPIPS + FPS from a single scene.")
  parser.add_argument("--scene_args_config_path", default="./params", help="Path to the scene arguments configuration YAML file.") 
  parser.add_argument("--method", default="3dgs", help="Method to use for evaluation. Options: '3dgs', 'ref-gs', 'rtr-gs', 'gs-ir'.")  
  parser.add_argument("--model_path", default=r"E:\\output\\ours\\eval",
                      help="scene model path")
  parser.add_argument("--json_name", default="collected_results.json",
                      help="Output JSON filename (written inside model_path)")
  args, _ = parser.parse_known_args()
  
  params = load_scene_args(args.scene_args_config_path, args.method)
  
  model_path = Path(args.model_path)
  out_file = model_path / args.json_name

  collected_results = {}

  results_path = model_path / "results.json"
  collected_results = parse_results_json(results_path)

  fps_path = model_path / "fps.txt"
  fps_value, count_value = parse_fps_txt(fps_path)
  collected_results["FPS"] = fps_value
  collected_results["count"] = count_value

  mae_path = model_path / "mae.txt"
  if mae_path.exists():
    with open(mae_path, "r", encoding="utf-8") as f:
      mae_value = f.readline().strip()
    collected_results["MAE"] = mae_value

  if params["method"].get("multi_stage", False):
    training_times = collect_multi_stage_training_times(model_path, params)
    collected_results["training_time"] = training_times["total"]
    collected_results["training_stages"] = training_times["stages"]

    usage_info = collect_multi_stage_usage(model_path, params)
    collected_results["peak_ram"] = usage_info["max_peak_ram"]
    collected_results["peak_vram"] = usage_info["max_peak_vram"]
    collected_results["training_stages_usage"] = usage_info["stages"]
  else:
    time_value = read_latest_training_time(model_path)
    collected_results["training_time"] = time_value

    usage_path = model_path / "usage.csv"
    peak_ram, peak_vram, mean_ram, mean_vram = parse_usage(usage_path)
    collected_results["peak_ram"] = peak_ram
    collected_results["peak_vram"] = peak_vram
    collected_results["mean_ram"] = mean_ram
    collected_results["mean_vram"] = mean_vram

  with open(out_file, 'w') as fp:
    json.dump(collected_results, fp, indent=True)

if __name__ == "__main__":
  main()