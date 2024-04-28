import logging
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)


def run_command(command, cwd=Path.cwd()):
    return subprocess.run(command, shell=True, stderr=sys.stdout, stdout=sys.stdout, cwd=cwd)


def move_file(source_file, destination_dir):
    try:
        destination_file = destination_dir / source_file.name
        source_file.rename(destination_file)
    except FileExistsError:
        log.warning(
            f"File '{source_file.name}' already exists in destination directory {destination_dir}. Did not move file")
    except Exception as e:
        print(f"An error occurred during move: {e}")
