import os
from argparse import ArgumentParser
from pathlib import Path
import time
import psutil
import shlex

from utils.context_utils import cd
from utils.parser_utils import load_datasets, load_scene_args
from monitor_utils import get_vram_procs, monitor
from process_utils import ProcessManager

def get_scene_config(scene: Path, dataset):
  args = ""
  if "scene_config" in dataset and scene.parent.name == dataset["id"]:
    if scene.name in dataset["scene_config"]:
      for param, value in dataset["scene_config"][scene.name].items():
        args += f" --{param} {value}"
  return args

def build_stage_args(params_stages, stage, dataset=None, scene=None):
  args = ""
  if stage in params_stages:
    if scene is not None and dataset is not None:
      args += get_scene_config(scene, dataset)

    if dataset is not None and dataset.get("white_background", False):
      args += " --white_background"

    stage_config = params_stages[stage]
    if type(stage_config["args"]) is str:
      args += stage_config["args"]
    else:
      args += stage_config["args"].get("base", "")
      if dataset is not None:
        if dataset.get("type", "") == "real":
          args += stage_config["args"].get("real", "")
        elif dataset.get("type", "") == "synthetic":
          args += stage_config["args"].get("synthetic", "")

    if scene is not None:
      scene_name = scene.parent.name + "/" + scene.name
      if "scene_overrides" in stage_config:
        args += stage_config["scene_overrides"].get(scene_name, "")

  return args

def get_scene_output_path(args, eval_dir, scene, repeat):
  scene_name = scene.parent.name + "/" + scene.name
  output_path = Path(eval_dir, scene_name)
  if args.repeats > 1:
    output_path = Path(eval_dir, f"{scene_name}_run_{repeat + 1}")
  return output_path


################
#   TRAINING   #
################
def training(args, eval_dir, scene, datasets, params, repeat=0):
  if params["method"].get("multi_stage", False):
    training_multistage(args, eval_dir, scene, datasets, params, repeat)
  else:
    training_single(args, eval_dir, scene, datasets, params, repeat)

def training_single(args, eval_dir: Path, scene: Path, dataset, params, repeat=0):
  print("Starting training for scene:", scene.name)

  python = params["method"]["python"]
  training_script = params["stages"]["training"].get("script", "train.py")
  training_args = build_stage_args(params["stages"], "training", dataset, scene)
  output_path = get_scene_output_path(args, eval_dir, scene, repeat)

  if (output_path / "point_cloud").exists():
    print(f"Output for {scene.parent.name + '/' + scene.name} already exists. Skipping training.")
    return

  working_dir = params["method"]["working_directory"]
  train_cmd = f"{python} {training_script} -s {scene} -m {output_path} {training_args}"
  
  if args.dry_run:
    print("Dry run enabled. Command that would be executed:")
    print(train_cmd)
    return

  output_path.mkdir(parents=True, exist_ok=True)
  with open(os.path.join(output_path, "commands.sh"), 'w') as file:
    file.write(train_cmd+ "\n")

  pm = ProcessManager()
  pm.register_signal_handlers()

  active_gpu_procs = get_vram_procs()

  scene_time = time.time()
  process = psutil.Popen(shlex.split(train_cmd, posix=False), cwd=working_dir, shell=False)
  pm.process = process
  pm.start_monitor(monitor, process.pid, active_gpu_procs, 1.0, os.path.join(output_path, "usage.csv"))
  try:
    process.wait()
  finally:
    pm.cleanup()
  scene_time= (time.time() - scene_time)/60.0

  timing_name = "training_time_" + time.strftime("%Y%m%d-%H%M%S") + ".txt"
  with open(os.path.join(output_path, timing_name), 'w') as file:
    file.write(str(scene_time)+ "\n")

def training_multistage(args, eval_dir, scene, dataset, params, repeat=0):
  print("Starting training for scene:", scene.name)

  python = params["method"]["python"]
  stages = params["stages"]["training"].keys()
  for stage in stages:
    training_script = params["stages"]["training"][stage].get("script", "train.py")
    training_args = build_stage_args(params["stages"]["training"], stage, dataset, scene)
    output_path = get_scene_output_path(args, eval_dir, scene, repeat)

    training_args = training_args.format(model_path=output_path)
    if "output_path_template" in params["stages"]["training"][stage]:
      template = params["stages"]["training"][stage]["output_path_template"]
      stage_output_path = Path(template.format(model_path=output_path))
    else:
      stage_output_path = output_path

    no_extra_args = params["stages"]["training"][stage].get("no_extra_args", False)

    if (stage_output_path / "point_cloud").exists() and not no_extra_args:
      print(f"Output for {scene.parent.name + '/' + scene.name} already exists. Skipping training.")
      continue

    working_dir = params["method"]["working_directory"]
    if no_extra_args:
      train_cmd = f"{python} {training_script} {training_args}"
    else:
      train_cmd = f"{python} {training_script} -s {scene} -m {stage_output_path} {training_args}"
    
    if args.dry_run:
      print("Dry run enabled. Command that would be executed:")
      print(train_cmd)
      continue

    output_path.mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output_path, "commands.sh"), 'a') as file:
      file.write(train_cmd+ "\n")

    pm = ProcessManager()
    pm.register_signal_handlers()

    active_gpu_procs = get_vram_procs()

    scene_time = time.time()
    process = psutil.Popen(shlex.split(train_cmd, posix=False), cwd=working_dir, shell=False)
    pm.process = process
    pm.start_monitor(monitor, process.pid, active_gpu_procs, 1.0, os.path.join(output_path, f"usage_{stage}.csv"))
    try:
      process.wait()
    finally:
      pm.cleanup()
    scene_time= (time.time() - scene_time)/60.0

    timing_name = f"training_time_{stage}_" + time.strftime("%Y%m%d-%H%M%S") + ".txt"
    with open(os.path.join(output_path, timing_name), 'w') as file:
      file.write(str(scene_time)+ "\n")

#################
#   RENDERING   #
#################
def rendering(args, eval_dir, scene, dataset, params, repeat=0):
  print("Starting rendering for scene:", scene)

  python = params["method"]["python"]
  rendering_script = params["stages"].get("rendering", {}).get("script", "train.py")
  rendering_args = build_stage_args(params["stages"], "rendering", dataset, scene)
  if dataset.get("has_normals", False) and params["method"].get("evaluate_normal_mae", False):
    rendering_args += " --save_normals"

  output_path = get_scene_output_path(args, eval_dir, scene, repeat)
  if "output_path_template" in params["stages"]["rendering"]:
    template = params["stages"]["rendering"]["output_path_template"]
    output_path = Path(template.format(model_path=output_path))

  rendering_args = rendering_args.format(model_path=output_path)

  saves_as_checkpoint = params["method"].get("saves_as_checkpoint", False)
  if not saves_as_checkpoint and not (output_path / "point_cloud").exists() and not args.dry_run:
    print(f"Output for {scene.parent.name + '/' + scene.name} does not exist. Skipping rendering.")
    return
  
  working_dir = params["method"]["working_directory"]
  render_cmd = f"{python} {rendering_script} -s {scene} -m {output_path} {rendering_args}"
  
  if args.dry_run:
    print("Dry run enabled. Command that would be executed:")
    print(render_cmd)
    return
  
  with open(os.path.join(output_path, "commands.sh"), 'a') as file:
    file.write(render_cmd + "\n")

  with cd(working_dir):
    os.system(render_cmd)


##########################
#   METRICS EVALUATION   #
##########################
def metrics_evaluation(args, eval_dir, scene, dataset, params, repeat=0):
  print("Starting metrics evaluation for scene:", scene)
  
  python = params["method"]["python"]
  metrics_script = params["stages"].get("metrics_evaluation", {}).get("script", "metrics.py")
  metrics_args = build_stage_args(params["stages"], "metrics_evaluation")
  
  output_path = get_scene_output_path(args, eval_dir, scene, repeat)
  if "output_path_template" in params["stages"]["metrics_evaluation"]:
    template = params["stages"]["metrics_evaluation"]["output_path_template"]
    output_path = Path(template.format(model_path=output_path))

  saves_as_checkpoint = params["method"].get("saves_as_checkpoint", False)
  if not saves_as_checkpoint and not (output_path / "point_cloud").exists() and not args.dry_run:
    print(f"Output for {scene.parent.name + '/' + scene.name} does not exist. Skipping metrics evaluation.")
    return

  working_dir = params["method"]["working_directory"]
  metrics_cmd = f"{python} {metrics_script} --model_paths {output_path} {metrics_args}"

  if args.dry_run:
    print("Dry run enabled. Command that would be executed:")
    print(metrics_cmd)
    return
  else:
    with cd(working_dir):
      os.system(metrics_cmd)


######################
#   MAE EVALUATION   #
######################
def mae_evaluation(args, eval_dir, scene, dataset, params, repeat=0):
  print("Starting MAE evaluation for scene:", scene)
  if not params["method"]["evaluate_normal_mae"]:
    return

  python = params["method"]["python"]
  mae_script = params["stages"].get("mae_evaluation", {}).get("script", "eval_mae.py")
  mae_args = build_stage_args(params["stages"], "mae_evaluation")
  
  output_path = get_scene_output_path(args, eval_dir, scene, repeat)
  if "output_path_template" in params["stages"]["mae_evaluation"]:
    template = params["stages"]["mae_evaluation"]["output_path_template"]
    output_path = Path(template.format(model_path=output_path))

  saves_as_checkpoint = params["method"].get("saves_as_checkpoint", False)
  if not saves_as_checkpoint and not (output_path / "point_cloud").exists() and not args.dry_run:
    print(f"Output for {scene.parent.name + '/' + scene.name} does not exist. Skipping MAE evaluation.")
    return

  working_dir = params["method"]["working_directory"]
  mae_cmd = f"{python} {mae_script} --source_path {scene} --model_path {output_path} {mae_args}"

  if args.dry_run:
    print("Dry run enabled. Command that would be executed:")
    print(mae_cmd)
    return
  else:
    with cd(working_dir):
      os.system(mae_cmd)


######################
#   FPS EVALUATION   #
######################
def fps_evaluation(args, eval_dir, scene, dataset, params, repeat=0):
  print("Starting FPS evaluation for scene:", scene)
  python = params["method"]["python"]
  fps_script = params["stages"].get("fps_evaluation", {}).get("script", "eval_fps.py")
  fps_args = build_stage_args(params["stages"], "fps_evaluation", dataset, scene)
  
  output_path = get_scene_output_path(args, eval_dir, scene, repeat)
  if "output_path_template" in params["stages"]["fps_evaluation"]:
    template = params["stages"]["fps_evaluation"]["output_path_template"]
    output_path = Path(template.format(model_path=output_path))

  fps_args = fps_args.format(model_path=output_path)
  
  saves_as_checkpoint = params["method"].get("saves_as_checkpoint", False)
  if not saves_as_checkpoint and not (output_path / "point_cloud").exists() and not args.dry_run:
    print(f"Output for {scene.parent.name + '/' + scene.name} does not exist. Skipping FPS evaluation.")
    return

  working_dir = params["method"]["working_directory"]
  fps_cmd = f"{python} {fps_script} -m {output_path} {fps_args}"

  if args.dry_run:
    print("Dry run enabled. Command that would be executed:")
    print(fps_cmd)
    return
  else:
    with cd(working_dir):
      os.system(fps_cmd)


#####################
#   RENDER VIDEOS   #
#####################
def render_videos(args, eval_dir, scene, dataset, params, repeat=0):
  print("Starting render videos for scene:", scene)
  python = params["method"]["python"]
  video_script = params["stages"]["render_videos"].get("script", "render-videos.py")
  video_args = build_stage_args(params["stages"], "render_videos", dataset, scene)
  
  output_path = get_scene_output_path(args, eval_dir, scene, repeat)
  if "output_path_template" in params["stages"]["render_videos"]:
    template = params["stages"]["render_videos"]["output_path_template"]
    output_path = Path(template.format(model_path=output_path))

  saves_as_checkpoint = params["method"].get("saves_as_checkpoint", False)
  if not saves_as_checkpoint and not (output_path / "point_cloud").exists() and not args.dry_run:
    print(f"Output for {scene.parent.name + '/' + scene.name} does not exist. Skipping FPS evaluation.")
    return

  working_dir = params["method"]["working_directory"]
  video_cmd = f"{python} {video_script} -m {output_path} {video_args}"

  if args.dry_run:
    print("Dry run enabled. Command that would be executed:")
    print(video_cmd)
    return
  else:
    with cd(working_dir):
      os.system(video_cmd)


##################
#   COLLECTING   #
##################
def collect_results(args, eval_dir, scene, dataset, params, repeat=0):
  print("Collecting results for scene:", scene)
  model_path = get_scene_output_path(args, eval_dir, scene, repeat)
  if "output_path_template" in params["stages"]["collect_results"]:
    template = params["stages"]["collect_results"]["output_path_template"]
    model_path = Path(template.format(model_path=model_path))

  collect_cmd = f"python collect.py --model_path {model_path} --method {params['method']['id']}"
  os.system(collect_cmd)

  print("Collecting results in:", eval_dir)
  collect_all_cmd = f"python collect_results.py --tsv --output_path {eval_dir}"
  os.system(collect_all_cmd)


def pipeline(args):
  datasets = load_datasets(args.datasets_config)
  params = load_scene_args(args.scene_args_config_path, args.method)

  if args.datasets:
    datasets = {k: v for k, v in datasets.items() if k in args.datasets}
    if not datasets:
      print("No valid datasets found for the provided dataset IDs.")
      exit(1)

  eval_dir = Path(args.output_dir) if args.output_dir else Path(params["method"]["output_path"], "eval_" + time.strftime("%Y%m%d-%H%M%S"))

  stages = args.stages if args.stages else ["training", "rendering", "metrics", "mae_eval", "fps", "collect"]

  for dataset_id, dataset in datasets.items():
    if args.real_scenes_only and not dataset.get("real", False):
      continue
    if args.synthetic_scenes_only and dataset.get("real", False):
      continue

    evaluate_normal_mae = dataset.get("has_normals", False) and params["method"].get("evaluate_normal_mae", True)

    scenes = dataset["scenes"]
    print(f"Evaluating dataset: {dataset_id} with {len(scenes)} scenes.")

    if args.filter_scenes_by:
      scenes = [scene for scene in scenes if scene.name in args.filter_scenes_by]
      print(f"Filtered scenes to {len(scenes)} based on provided filter: {args.filter_scenes_by}")

    for scene in scenes:
      if args.repeats == 1:
        print(f"Evaluating scene: {scene}")
      for repeat in range(args.repeats):
        if args.repeats > 1:
          print(f"Repeating evaluation for scene: {scene}, repeat: {repeat + 1}/{args.repeats}")

        if "training" in stages:
          training(args, eval_dir, scene, dataset, params, repeat)

        if "rendering" in stages:
          rendering(args, eval_dir, scene, dataset, params, repeat)
        
        if "metrics" in stages:
          metrics_evaluation(args, eval_dir, scene, dataset, params, repeat)

        if evaluate_normal_mae and "mae_eval" in stages:
          mae_evaluation(args, eval_dir, scene, dataset, params, repeat)

        if "fps" in stages:
          fps_evaluation(args, eval_dir, scene, dataset, params, repeat)

        if "render_videos" in stages:
          render_videos(args, eval_dir, scene, dataset, params, repeat)

        if "collect" in stages:
          collect_results(args, eval_dir, scene, dataset, params, repeat)

  print("Done with full evaluation for all scenes!")

if __name__ == "__main__":
  parser = ArgumentParser(description="Full evaluation script parameters")
  parser.add_argument("--datasets_config", default="./params/datasets.yaml", help="Path to the datasets configuration YAML file.")
  parser.add_argument("--scene_args_config_path", default="./params", help="Path to the scene arguments configuration YAML file.") 
  parser.add_argument("--method", default="3dgs", help="Method to use for evaluation. Options: '3dgs', 'ref-gs', 'rtr-gs', 'gs-ir'.")
  parser.add_argument("--output_dir", default=None)
  parser.add_argument("--repeats", default=1, type=int, help="How many times to repeat the evaluation for each scene. Useful for averaging results over multiple runs.")
  parser.add_argument("--stages", nargs='+', default=["training", "rendering", "metrics", "mae_eval", "fps", "collect"], help="Stages to run. Options: 'training', 'rendering', 'metrics', 'mae_eval', 'fps', 'render_videos', 'collect'.")
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