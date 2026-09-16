import yaml
from pathlib import Path

# Load datasets and parameters
def load_datasets(config_path):
  with open(config_path, encoding="utf-8") as file:
    config = yaml.safe_load(file)
  data_config = config["data"]
  base_path = Path(data_config["base_path"])
  datasets = {}
  for dataset_id, dataset_config in data_config["datasets"].items():
    directory = dataset_config.get("directory", dataset_id)
    dataset_path = base_path / directory
    datasets[dataset_id] = {
      **dataset_config,
      "id": dataset_id,
      "path": dataset_path,
      "scenes": sorted(
        path
        for path in dataset_path.iterdir()
        if path.is_dir()
      ),
    }
  return datasets

def load_scene_args(config_path, method):
  args_path = Path(f"{config_path}/scene_args_{method}.yaml")
  if not args_path.exists():
    raise FileNotFoundError(f"Scene args config file not found: {args_path}")
  
  with open(args_path, encoding="utf-8") as file:
    config = yaml.safe_load(file)
  return config