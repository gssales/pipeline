

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

from monitor_utils import monitor
from process_utils import ProcessManager

pm = ProcessManager()
pm.register_signal_handlers()

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
def training(args, eval_dir, scenes, datasets, parameters):
  print("Starting training for all scenes...")
  common_args = parameters["parameters"]["training"]["base"]

  scene_times = {}
  progress = tqdm(total=len(scenes), position=1) 
  for scene in scenes:
    progress.set_description(f"Training {scene.parent.name}/{scene.name}")
    
    dataset = scene.parent.name
    train_args = get_dataset_args(dataset, "training", datasets, parameters) + common_args
    dataset_scene = scene.parent.name + "/" + scene.name

    output_path = Path(parameters["base_path"], eval_dir, dataset_scene)
    train_command = f"{parameters['conda_env']}/python train.py -s {scene} -m {output_path} {train_args}"

    if args.dry_run:
      print("Dry run enabled. Command that would be executed:")
      print(train_command)
      continue

    output_path.mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output_path, "commands.sh"), 'w') as file:
      file.write(train_command+ "\n")

    print("")
    scene_time = time.time()
    process = psutil.Popen(shlex.split(train_command), cwd=parameters["script_path"], shell=False)
    pm.process = process
    pm.start_monitor(monitor, process.pid, 1.0, os.path.join(output_path, "usage.csv"))
    try:
      process.wait()
    finally:
      pm.cleanup()
    scene_times[dataset_scene] = (time.time() - scene_time)/60.0

    progress.update(1)
  progress.close()
  with open(os.path.join(parameters["base_path"], eval_dir, "timing.json"), 'w') as file:
    json.dump(scene_times, file, indent=True)



#################
#   RENDERING   #
#################
def rendering(args, eval_dir, scenes, datasets, parameters):
  print("Starting rendering for all scenes...")
  common_args = parameters["parameters"]["rendering"]["base"]

  progress = tqdm(total=len(scenes), position=1) 
  for scene in scenes:
    progress.set_description(f"Rendering {scene.parent.name}/{scene.name}")
    
    dataset = scene.parent.name
    render_args = get_dataset_args(dataset, "rendering", datasets, parameters) + common_args
      
    dataset_scene = scene.parent.name + "/" + scene.name
    output_path = Path(parameters["base_path"], eval_dir, dataset_scene)
    render_command = f"{parameters['conda_env']}/python render.py -s {scene} -m {output_path} {render_args}"
    
    if args.dry_run:
      print("Dry run enabled. Command that would be executed:")
      print(render_command)
      continue

    with open(os.path.join(output_path, "commands.sh"), 'a') as file:
      file.write(render_command + "\n")

    with cd(parameters["script_path"]):
      os.system(render_command)
    progress.update(1)
  progress.close()



######################
#   MAE EVALUATION   #
######################
def mae_evaluation(args, eval_dir, scenes, parameters):
  print("Starting MAE evaluation for specified datasets...")
  mae_scenes = [scene for scene in scenes if scene.parent.name in parameters["mae_eval_datasets"]] 

  progress = tqdm(total=len(mae_scenes), position=1) 
  for scene in mae_scenes:
    dataset = scene.parent.name
    if dataset not in parameters["mae_eval_datasets"]:
      continue

    progress.set_description(f"MAE Evaluation {scene.parent.name}/{scene.name}")
    
    dataset_scene = scene.parent.name + "/" + scene.name
    output_path = Path(parameters["base_path"], eval_dir, dataset_scene, "test")
    mae_command = f"{parameters['conda_env']}/python eval_mae.py --gt_path {scene} --render_path {output_path}"
  
    if args.dry_run:
      print("Dry run enabled. Command that would be executed:")
      print(mae_command)
      continue

    with open(os.path.join(output_path, "commands.sh"), 'a') as file:
      file.write(mae_command + "\n")

    with cd(parameters["script_path"]):
      os.system(mae_command)
    progress.update(1)
  progress.close()
  

######################
#   FPS EVALUATION   #
######################
def fps_evaluation(args, eval_dir, scenes, parameters):
  print("Starting FPS evaluation for all scenes...")

  progress = tqdm(total=len(scenes), position=1)
  for scene in scenes:
    progress.set_description(f"FPS Evaluation {scene.parent.name}/{scene.name}")

    dataset_scene = scene.parent.name + "/" + scene.name
    output_path = Path(parameters["base_path"], eval_dir, dataset_scene)
    fps_command = f"{parameters['conda_env']}/python eval_fps.py -s {scene} -m {output_path}"
    
    if args.dry_run:
      print("Dry run enabled. Command that would be executed:")
      print(fps_command)
      continue

    with open(os.path.join(output_path, "commands.sh"), 'a') as file:
      file.write(fps_command + "\n")
    
    with cd(parameters["script_path"]):
      os.system(fps_command)
    progress.update(1)
  progress.close()


##########################
#   METRICS EVALUATION   #
##########################
def metrics_evaluation(args, eval_dir, scenes, parameters):
  print("Starting metrics evaluation for all scenes...")

  model_paths = ""
  for scene in scenes:
    dataset_scene = scene.parent.name + "/" + scene.name
    output_path = Path(parameters["base_path"], eval_dir, dataset_scene)
    model_paths += f"{output_path} "

  metrics_command = f"{parameters['conda_env']}/python metrics.py -m {model_paths}"
  if args.dry_run:
    print("Dry run enabled. Command that would be executed:")
    print(metrics_command)
  else:
    with cd(parameters["script_path"]):
      os.system(metrics_command)


##################
#   COLLECTING   #
##################
def collect_results(args, eval_dir, parameters):
  output_path = Path(parameters["base_path"], eval_dir)
  print("Collecting results in:", output_path)
  collect_command = "python collect_results.py --tsv --output_path " + str(output_path)
  os.system(collect_command)

def render_videos(args, eval_dir, parameters):
  output_path = Path(parameters["base_path"], eval_dir)
  print("Rendering videos for all scenes in:", output_path)
  render_command = "python render_videos.py --input_path " + str(output_path)
  os.system(render_command)


def pipeline(args):
  
  eval_dir = args.output_dir if args.output_dir else "eval_" + time.strftime("%Y%m%d-%H%M%S")

  scenes, datasets = load_datasets(args)
  params = load_parameters(args.method)
  
  if not args.skip_training:
    training(args, eval_dir, scenes, datasets, params)

  if not args.skip_rendering:
    rendering(args, eval_dir, scenes, datasets, params)
  
  if not args.skip_fps:
    fps_evaluation(args, eval_dir, scenes, params)

  if not args.skip_metrics:
    metrics_evaluation(args, eval_dir, scenes, params)

  if params.get("mae_eval_datasets", False) and not args.skip_mae_eval:
    mae_evaluation(args, eval_dir, scenes, params)

  if not args.skip_collect_results:
    collect_results(args, eval_dir, params)

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
  parser.add_argument("--method", default="3dgs")
  args, _ = parser.parse_known_args()

  if args.real_scenes_only and args.synthetic_scenes_only:
    print("Cannot specify both --real_scenes_only and --synthetic_scenes_only. Please choose one or neither.")
    exit(1)

  pipeline(args)