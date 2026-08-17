

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

def is_synthetic_dataset(dataset, datasets):
  return dataset in datasets["data"]["synthetic_datasets"]

def is_real_dataset(dataset, datasets):
  return dataset in datasets["data"]["real_datasets"]

def get_dataset_args(dataset, stage, datasets, parameters):
  dataset_args = ""
  if stage not in parameters["parameters"]:
    return ""
  if "real" in parameters["parameters"][stage] and is_real_dataset(dataset, datasets):
    dataset_args += parameters["parameters"][stage]["real"]
  if "synthetic" in parameters["parameters"][stage] and is_synthetic_dataset(dataset, datasets):
    dataset_args += parameters["parameters"][stage]["synthetic"]
  return dataset_args

def get_training_stage_args(dataset, stage, datasets, parameters):
  stage_args = ""
  if stage not in parameters["parameters"]["training"]:
    return ""
  stage_args += parameters["parameters"]["training"][stage]["base"]
  if "real" in parameters["parameters"]["training"][stage] and is_real_dataset(dataset, datasets):
    stage_args += parameters["parameters"]["training"][stage]["real"]
  if "synthetic" in parameters["parameters"]["training"][stage] and is_synthetic_dataset(dataset, datasets):
    stage_args += parameters["parameters"]["training"][stage]["synthetic"]
  return stage_args

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
  common_args = parameters["parameters"]["training"]["base"]

  dataset = scene.parent.name
  train_args = get_dataset_args(dataset, "training", datasets, parameters) + common_args
  stage1_args = get_training_stage_args(dataset, "stage1", datasets, parameters)
  stage2_args = get_training_stage_args(dataset, "stage2", datasets, parameters)
  dataset_scene = scene.parent.name + "/" + scene.name
  # train_args += parameters["args"].get(dataset_scene, "")

  train_script = "train.py"
  if "train_script" in parameters and dataset in parameters["train_script"]:
    train_script = parameters["train_script"][dataset]
  
  for repeat in range(args.repeats):
    output_path = Path(eval_dir, dataset_scene)
    if args.repeats > 1:
      print(f"Repeat {repeat+1}/{args.repeats} for scene: {scene}")
      output_path = Path(eval_dir, f"{dataset_scene}_run_{repeat+1}")

    if (output_path / "stage1").exists():
      print(f"Output for {dataset_scene} already exists. Skipping training.")
      continue

    stage1_output_path = output_path / "stage1"
    train1_command = f"{parameters['conda_env']}/python {train_script} -s {scene} -m {stage1_output_path} {train_args} {stage1_args}"
    checkpoint_path = stage1_output_path / "checkpoint" / "chkpnt30000.pth"
    baking_command = f"{parameters['conda_env']}/python baking.py --checkpoint {checkpoint_path} --bound 2.0 --occlu_res 128"
    stage2_output_path = output_path / "stage2"
    occlusion_path = stage1_output_path / "checkpoint" / "occlusion_volumes.pth"
    train2_command = f"{parameters['conda_env']}/python {train_script} -s {scene} -m {stage2_output_path} -c {checkpoint_path} --occlusion_path {occlusion_path} {train_args} {stage2_args}"
    if is_synthetic_dataset(dataset, datasets):
      stage2_checkpoint_path = stage2_output_path / "checkpoint" / "chkpnt40000.pth"
      render_eval_command = f"{parameters['conda_env']}/python render_and_eval.py -m {stage2_output_path} -c {stage2_checkpoint_path} --occlusion_path {occlusion_path} --ref_map --compute_with_prt --metallic -t render_ref_pbr"

    if args.dry_run:
      print("Dry run enabled. Command that would be executed:")
      print(train1_command)
      print(baking_command)
      print(train2_command)
      continue

    output_path.mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output_path, "commands.sh"), 'w') as file:
      file.write(train1_command+ "\n")
      file.write(baking_command+ "\n")
      file.write(train2_command+ "\n")

    pm = ProcessManager()
    pm.register_signal_handlers()

    active_gpu_procs = get_vram_procs()

    scene_times = {}
    scene_time = time.time()
    # stage 1 training
    process = psutil.Popen(shlex.split(train1_command), cwd=parameters["script_path"], shell=False)
    pm.process = process
    pm.start_monitor(monitor, process.pid, active_gpu_procs, 1.0, os.path.join(output_path, "usage_stage1.csv"))
    try:
      process.wait()
    finally:
      pm.cleanup()

    # baking
    pm = ProcessManager()
    pm.register_signal_handlers()

    active_gpu_procs = get_vram_procs()
    process = psutil.Popen(shlex.split(baking_command), cwd=parameters["script_path"], shell=False)
    pm.process = process
    pm.start_monitor(monitor, process.pid, active_gpu_procs, 1.0, os.path.join(output_path, "usage_baking.csv"))
    try:
      process.wait()
    finally:
      pm.cleanup()

    # stage 2 training
    pm = ProcessManager()
    pm.register_signal_handlers()

    active_gpu_procs = get_vram_procs()
    process = psutil.Popen(shlex.split(train2_command), cwd=parameters["script_path"], shell=False)
    pm.process = process
    pm.start_monitor(monitor, process.pid, active_gpu_procs, 1.0, os.path.join(output_path, "usage_stage2.csv"))
    try:
      process.wait()
    finally:
      pm.cleanup()

    scene_times[dataset_scene] = (time.time() - scene_time)/60.0

    if is_synthetic_dataset(dataset, datasets):
      with cd(parameters["script_path"]):
        os.system(render_eval_command)

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
  print("Starting FPS evaluation for scene:", scene)
  common_args = parameters["parameters"]["training"]["base"]
  
  dataset = scene.parent.name
  train_args = get_dataset_args(dataset, "training", datasets, parameters) + common_args
  stage2_args = get_training_stage_args(dataset, "stage2", datasets, parameters)
  dataset_scene = scene.parent.name + "/" + scene.name

  fps_script = "eval_fps.py"

  for repeat in range(args.repeats):
    output_path = Path(eval_dir, dataset_scene)
    if args.repeats > 1:
      print(f"Repeat {repeat+1}/{args.repeats} for scene: {scene}")
      output_path = Path(eval_dir, f"{dataset_scene}_run_{repeat+1}")

    if not (output_path / "stage2").exists():
      print(f"Output for {dataset_scene} does not exist. Skipping FPS evaluation.")
      continue

    fps_args = f"{train_args} {stage2_args}"
    eval_path = Path(output_path, "stage2")
    if not eval_path.exists():
      print(f"Stage 2 output for {dataset_scene} does not exist. Skipping FPS evaluation.")
      continue

    stage1_output_path = output_path / "stage1"
    checkpoint_path = stage1_output_path / "checkpoint" / "chkpnt30000.pth"
    occlusion_path = stage1_output_path / "checkpoint" / "occlusion_volumes.pth"
    fps_command = f"{parameters['conda_env']}/python {fps_script} -s {scene} -m {eval_path} -c {checkpoint_path} --occlusion_path {occlusion_path} {fps_args}"
  
    if args.dry_run:
      print("Dry run enabled. Command that would be executed:")
      print(fps_command)
      return
  
    with cd(parameters["script_path"]):
      os.system(fps_command)


##########################
#   METRICS EVALUATION   #
##########################
def metrics_evaluation(args, eval_dir, scene, parameters):
  pass


##################
#   COLLECTING   #
##################
def collect_results(output_path):
  print("Collecting results in:", output_path)
  collect_command = "python collect_results_rtr-gs.py --tsv --output_path " + str(output_path)
  os.system(collect_command)

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
    
    if not args.skip_fps:
      fps_evaluation(args, eval_dir, scene, datasets, params)

    # if not args.skip_metrics:
    #   metrics_evaluation(args, eval_dir, scene, params)

    # if params.get("mae_eval_datasets", False) and not args.skip_mae_eval:
    #   mae_evaluation(args, eval_dir, scene, params)

    if not args.skip_collect_results:
      collect_results(eval_dir)

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
  parser.add_argument("--method", default="rtr-gs")
  parser.add_argument("--repeats", default=1, type=int, help="How many times to repeat the evaluation for each scene. Useful for averaging results over multiple runs.")
  args, _ = parser.parse_known_args()

  if args.real_scenes_only and args.synthetic_scenes_only:
    print("Cannot specify both --real_scenes_only and --synthetic_scenes_only. Please choose one or neither.")
    exit(1)

  pipeline(args)