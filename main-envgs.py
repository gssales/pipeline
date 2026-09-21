

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
from utils.context_utils import cd
from utils.parser_utils import load_datasets, load_scene_args


################
#   TRAINING   #
################
def training(args, dataset_scene, scene, datasets, parameters):
  print("Starting training for scene:", scene)
  train_command = f"{parameters['conda_env']}/evc-train -c configs/exps/envgs/{scene.parent.name}/envgs_{scene.name}.yaml exp_name={dataset_scene}"

  output_path = os.path.join('/mnt/output/envgs/record', dataset_scene)

  if args.dry_run:
    print("Dry run enabled. Command that would be executed:")
    print(train_command)
    return

  pm = ProcessManager()
  pm.register_signal_handlers()

  active_gpu_procs = get_vram_procs()

  scene_times = {}
  scene_time = time.time()
  process = psutil.Popen(shlex.split(train_command), cwd=parameters["working_directory"], shell=False)
  pm.process = process
  pm.start_monitor(monitor, process.pid, active_gpu_procs, 10.0, os.path.join(output_path, "usage.csv"))
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
def rendering(args, dataset_scene, scene, datasets, params):
  print("Starting rendering for scene:", scene)

  python = params["method"]["python"]
  rendering_script = "eval.py"
  
  working_dir = params["method"]["working_directory"]
  config = f"configs/exps/envgs/{scene.parent.name}/envgs_{scene.name}.yaml"
  checkpoint = f"{dataset_scene}/latest.pt"
  model_path = f"/mnt/output/envgs/record/{dataset_scene}/"
  render_cmd = f"{python} {rendering_script} --config {config} --checkpoint {checkpoint} --model-path {model_path} --mode render --normal-space world"
  
  if args.dry_run:
    print("Dry run enabled. Command that would be executed:")
    print(render_cmd)
    return
  
  with open(os.path.join(model_path, "commands.sh"), 'a') as file:
    file.write(render_cmd + "\n")

  with cd(working_dir):
    os.system(render_cmd)



######################
#   MAE EVALUATION   #
######################
def mae_evaluation(args, eval_dir, scene, parameters):
  pass
  

######################
#   FPS EVALUATION   #
######################
def fps_evaluation(args, dataset_scene, scene, datasets, params):
  print("Starting rendering for scene:", scene)
  
  python = params["method"]["python"]
  fps_script = "eval.py"
  
  working_dir = params["method"]["working_directory"]
  config = f"configs/exps/envgs/{scene.parent.name}/envgs_{scene.name}.yaml"
  checkpoint = f"{dataset_scene}/latest.pt"
  model_path = f"/mnt/output/envgs/record/{dataset_scene}/"
  fps_cmd = f"{python} {fps_script} --config {config} --checkpoint {checkpoint} --model-path {model_path} --mode fps --repeats 100"
  
  if args.dry_run:
    print("Dry run enabled. Command that would be executed:")
    print(fps_cmd)
    return
  
  with open(os.path.join(model_path, "commands.sh"), 'a') as file:
    file.write(fps_cmd + "\n")

  with cd(working_dir):
    os.system(fps_cmd)


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
  
  datasets = load_datasets(args.datasets_config)
  params = load_scene_args(args.scene_args_config_path, args.method)
  
  if args.datasets:
    datasets = {k: v for k, v in datasets.items() if k in args.datasets}
    if not datasets:
      print("No valid datasets found for the provided dataset IDs.")
      exit(1)

  eval_dir = "" #Path(args.output_dir) if args.output_dir else Path(params["base_path"], "eval_" + time.strftime("%Y%m%d-%H%M%S"))

  stages = args.stages if args.stages else ["training"]

  for dataset_id, dataset in datasets.items():
    if args.real_scenes_only and not dataset.get("real", False):
      continue
    if args.synthetic_scenes_only and dataset.get("real", False):
      continue
    
    evaluate_normal_mae = dataset.get("has_normals", False) and params["method"].get("evaluate_normal_mae", True)

    scenes = dataset["scenes"]
    print(f"Evaluating dataset: {dataset_id} with {len(scenes)} scenes.")
    
    dataset_scene = scene.parent.name + "_" + scene.name + "_" + time.strftime("%Y%m%d-%H%M%S")

    if args.filter_scenes_by:
      scenes = [scene for scene in scenes if scene.name in args.filter_scenes_by]
      print(f"Filtered scenes to {len(scenes)} based on provided filter: {args.filter_scenes_by}")

    for scene in scenes:
      if "training" in stages:
        training(args, dataset_scene, scene, datasets, params)

      if not args.skip_rendering:
        rendering(args, dataset_scene, scene, datasets, params)
      
      if not args.skip_fps:
        fps_evaluation(args, dataset_scene, scene, datasets, params)

      # if not args.skip_metrics:
      #   metrics_evaluation(args, eval_dir, scene, params)

      # if params.get("mae_eval_datasets", False) and not args.skip_mae_eval:
      #   mae_evaluation(args, eval_dir, scene, params)

      # if not args.skip_collect_results:
      #   collect_results(eval_dir)

  print("Done with full evaluation for all scenes!")

if __name__ == "__main__":
  parser = ArgumentParser(description="Full evaluation script parameters")
  parser.add_argument("--datasets_config", default="./params/datasets.yaml", help="Path to the datasets configuration YAML file.")
  parser.add_argument("--scene_args_config_path", default="./params", help="Path to the scene arguments configuration YAML file.") 
  parser.add_argument("--method", default="envgs")
  parser.add_argument("--output_dir", default=None)
  parser.add_argument("--stages", nargs='+', default=["training"], help="Stages to run. Options: 'training', 'rendering', 'metrics', 'mae_eval', 'fps', 'render_videos', 'collect'.")
  parser.add_argument('--real_scenes_only', action='store_true')
  parser.add_argument('--synthetic_scenes_only', action='store_true')
  parser.add_argument("--dry_run", action="store_true", help="If set, the script will print the commands that would be run without executing them.")
  parser.add_argument("--datasets", nargs='+', help="List of dataset IDs to filter and run the evaluation on. If not provided, all datasets will be evaluated.")
  parser.add_argument("--filter_scenes_by", nargs='+', default=[], help="List of scene names to filter and run the evaluation on. If not provided, all scenes will be evaluated.")
  args, _ = parser.parse_known_args()

  if args.real_scenes_only and args.synthetic_scenes_only:
    print("Cannot specify both --real_scenes_only and --synthetic_scenes_only. Please choose one or neither.")
    exit(1)

  pipeline(args)