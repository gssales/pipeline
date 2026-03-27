import psutil
import time
try:
  from pynvml import *
  nvmlInit()
  NVML_AVAILABLE = True
except Exception:
  NVML_AVAILABLE = False

def get_vram_procs():
  """Return list of pid for processes using VRAM on the same GPU as the given PID."""
  if not NVML_AVAILABLE:
    return []

  procs = []
  try:
    device_count = nvmlDeviceGetCount()
    for i in range(device_count):
      handle = nvmlDeviceGetHandleByIndex(i)

      # Running compute processes
      try:
        procs = nvmlDeviceGetComputeRunningProcesses(handle)
      except:
        procs = []

      # Some drivers require this too
      try:
        procs += nvmlDeviceGetGraphicsRunningProcesses(handle)
      except:
        pass

      for p in procs:
        procs.append(p.pid)

  except Exception:
      pass

  return procs

def get_vram_usage(active_pids):
  """Return VRAM usage in MB for a given PID (if available)."""
  if not NVML_AVAILABLE:
    return 0

  vram = 0
  try:
    device_count = nvmlDeviceGetCount()
    for i in range(device_count):
      handle = nvmlDeviceGetHandleByIndex(i)

      # Running compute processes
      try:
        procs = nvmlDeviceGetComputeRunningProcesses(handle)
      except:
        procs = []

      # Some drivers require this too
      try:
        procs += nvmlDeviceGetGraphicsRunningProcesses(handle)
      except:
        pass

      for p in procs:
        if p.pid not in active_pids:
          vram += p.usedGpuMemory / (1024 ** 2)

  except Exception:
      pass

  return vram


def monitor(pid, stop_event, vram_pids, interval=1.0, log_file=None):
  process = psutil.Process(pid)

  f = open(log_file, "w") if log_file else None
  if f:
    f.write("time,ram_mb,vram_mb\n")

  peak_ram = 0
  peak_vram = 0

  while not stop_event.is_set():
    if not process.is_running():
      break

    try:
      # RAM
      ram = process.memory_info().rss / (1024 ** 2)

      # VRAM
      vram = get_vram_usage(vram_pids)

      peak_ram = max(peak_ram, ram)
      peak_vram = max(peak_vram, vram)

      if f:
        f.write(f"{time.time()},{ram:.2f},{vram:.2f}\n")
        f.flush()

    except psutil.NoSuchProcess:
      break
    except Exception:
      pass

    time.sleep(interval)

  if f:
    f.close()

  return peak_ram, peak_vram