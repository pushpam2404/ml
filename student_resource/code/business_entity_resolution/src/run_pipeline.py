"""Single entry point: train then predict, with default paths matching the
challenge's `student_resource/` layout."""
import subprocess
import sys
import os

HERE = os.path.dirname(__file__)


def main():
    subprocess.run([sys.executable, os.path.join(HERE, "train.py"), *sys.argv[1:]], check=True)
    subprocess.run([sys.executable, os.path.join(HERE, "predict.py")], check=True)


if __name__ == "__main__":
    main()
