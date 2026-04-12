

# 0 - dataset list and parameters
# 1 - train
# 1.5 - during training, measure fps and memory usage
# 2 - measure fps
# 3 - render images
# 4 - measure metrics
# 5 - render videos

from contextlib import contextmanager
import json
import os
from argparse import ArgumentParser
from pathlib import Path
import time
import yaml
from tqdm import tqdm
import psutil
import shlex

from monitor_utils import get_vram_procs, monitor
from process_utils import ProcessManager

# Load datasets and parameters
def read_scenes(dataset_path: Path):
  scenes = []
  if dataset_path.exists() and dataset_path.is_dir():
    for scene in os.listdir(dataset_path):
      scene_path = dataset_path / scene
      if scene_path.exists() and scene_path.is_dir():
        scenes.append(scene_path)
  return scenes

def get_dataset_args(dataset, stage, datasets, parameters):
  dataset_args = ""
  if stage not in parameters["parameters"]:
    return ""
  if "real" in parameters["parameters"][stage] and dataset in datasets["data"]["real_datasets"]:
    dataset_args += parameters["parameters"][stage]["real"]
  if "synthetic" in parameters["parameters"][stage] and dataset in datasets["data"]["synthetic_datasets"]:
    dataset_args += parameters["parameters"][stage]["synthetic"]
  return dataset_args

def load_datasets(args):
  scenes = []
  datasets = {}
  with open("params/datasets.yaml", 'r') as file:
    try:
      datasets = yaml.safe_load(file)
      basePath = Path(datasets["data"]["base_path"])
      if not args.synthetic_scenes_only:
        for dataset in datasets["data"]["real_datasets"]:
          scenes.extend(read_scenes(basePath / dataset))
      if not args.real_scenes_only:
        for dataset in datasets["data"]["synthetic_datasets"]:
          scenes.extend(read_scenes(basePath / dataset))
        
    except yaml.YAMLError as exc:
      print(exc)
  return scenes, datasets

def load_parameters(method):
  params_path = Path(f"params/scene_args_{method}.yaml")
  if not params_path.exists():
    print(f"Parameters file {params_path} not found. Please make sure it exists and is named correctly.")
    exit(1)

  parameters = {}
  with open(params_path, 'r') as file:
    try:
      parameters = yaml.safe_load(file)
    except yaml.YAMLError as exc:
      print(exc)
  return parameters

@contextmanager
def cd(destination):
  old = os.getcwd()
  os.chdir(destination)
  try:
    yield
  finally:
    os.chdir(old)


################
#   TRAINING   #
################
def training(args, eval_dir, scene, datasets, parameters):
  print("Starting training for scene:", scene)
  dataset_scene = scene.parent.name + "/" + scene.name

  output_path = Path(eval_dir, dataset_scene)
  if (output_path / "RENDER").exists():
    print(f"Output for {dataset_scene} already exists. Skipping training.")
    return

  train_command = f"{parameters['conda_env']}/evc-train -c configs/exps/envgs/{scene.parent.name}/envgs_{scene.name}.yaml exp_name={output_path}"

  if args.dry_run:
    print("Dry run enabled. Command that would be executed:")
    print(train_command)
    return

  pm = ProcessManager()
  pm.register_signal_handlers()

  active_gpu_procs = get_vram_procs()

  scene_times = {}
  scene_time = time.time()
  process = psutil.Popen(shlex.split(train_command), cwd=parameters["script_path"], shell=False)
  pm.process = process
  pm.start_monitor(monitor, process.pid, active_gpu_procs, 1.0, os.path.join(output_path, "usage.csv"))
  try:
    process.wait()
  finally:
    pm.cleanup()
  scene_times[dataset_scene] = (time.time() - scene_time)/60.0

  timing_name = "timing_" + time.strftime("%Y%m%d-%H%M%S") + ".json"
  with open(os.path.join(output_path, timing_name), 'w') as file:
    json.dump(scene_times, file, indent=True)



#################
#   RENDERING   #
#################
def rendering(args, eval_dir, scene, datasets, parameters):
  pass



######################
#   MAE EVALUATION   #
######################
def mae_evaluation(args, eval_dir, scene, parameters):
  pass
  

######################
#   FPS EVALUATION   #
######################
def fps_evaluation(args, eval_dir, scene, datasets, parameters):
  pass


##########################
#   METRICS EVALUATION   #
##########################
def metrics_evaluation(args, eval_dir, scene, parameters):
  pass


##################
#   COLLECTING   #
##################
def collect_results(output_path):
  pass

def render_videos(args, eval_dir, parameters):
  pass


def pipeline(args):
  
  scenes, datasets = load_datasets(args)
  params = load_parameters(args.method)

  eval_dir = Path(args.output_dir) if args.output_dir else Path(params["base_path"], "eval_" + time.strftime("%Y%m%d-%H%M%S"))

  for scene in scenes:
    if not args.skip_training:
      training(args, eval_dir, scene, datasets, params)

    # if not args.skip_rendering:
    #   rendering(args, eval_dir, scene, datasets, params)
    
    # if not args.skip_fps:
    #   fps_evaluation(args, eval_dir, scene, datasets, params)

    # if not args.skip_metrics:
    #   metrics_evaluation(args, eval_dir, scene, params)

    # if params.get("mae_eval_datasets", False) and not args.skip_mae_eval:
    #   mae_evaluation(args, eval_dir, scene, params)

    # if not args.skip_collect_results:
    #   collect_results(eval_dir)

  print("Done with full evaluation for all scenes!")

if __name__ == "__main__":
  parser = ArgumentParser(description="Full evaluation script parameters")
  parser.add_argument("--skip_training", action="store_true")
  parser.add_argument("--skip_rendering", action="store_true")
  parser.add_argument("--skip_fps", action="store_true")
  parser.add_argument("--skip_metrics", action="store_true")
  parser.add_argument("--skip_collect_results", action="store_true")
  parser.add_argument("--skip_render_videos", action="store_true")
  parser.add_argument("--skip_mae_eval", action="store_true")
  parser.add_argument('--real_scenes_only', action='store_true')
  parser.add_argument('--synthetic_scenes_only', action='store_true')
  parser.add_argument("--dry_run", action="store_true", help="If set, the script will print the commands that would be run without executing them.")
  parser.add_argument("--output_dir", default=None)
  parser.add_argument("--method", default="envgs")
  args, _ = parser.parse_known_args()

  if args.real_scenes_only and args.synthetic_scenes_only:
    print("Cannot specify both --real_scenes_only and --synthetic_scenes_only. Please choose one or neither.")
    exit(1)

  pipeline(args)