from contextlib import contextmanager
import os

@contextmanager
def cd(destination):
  old = os.getcwd()
  os.chdir(destination)
  try:
    yield
  finally:
    os.chdir(old)