import threading
import signal
import sys
import subprocess

class ProcessManager:
  def __init__(self):
    self.process = None
    self.monitor_thread = None
    self.stop_event = threading.Event()

  def start_monitor(self, monitor_fn, pid, *args):
    self.monitor_thread = threading.Thread(
      target=monitor_fn,
      args=(pid, self.stop_event, *args),
      daemon=True
    )
    self.monitor_thread.start()

  def cleanup(self):
    print("\n[PIPELINE] Cleaning up...")

    # Stop monitor
    self.stop_event.set()

    # Terminate subprocess
    if self.process and self.process.poll() is None:
      print("[PIPELINE] Terminating subprocess...")
      self.process.terminate()

      try:
        self.process.wait(timeout=5)
      except subprocess.TimeoutExpired:
        print("[PIPELINE] Killing subprocess...")
        self.process.kill()

    # Join monitor thread
    if self.monitor_thread and self.monitor_thread.is_alive():
      self.monitor_thread.join(timeout=5)

    print("[PIPELINE] Done.")

  def register_signal_handlers(self):
    def handler(sig, frame):
      self.cleanup()
      sys.exit(1)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)