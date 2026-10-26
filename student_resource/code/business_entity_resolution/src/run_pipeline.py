"""Single entry point: train, then predict with the model that was just trained.

    python run_pipeline.py --data-dir ../../../dataset --model-out ../model --output-dir ../../../output

--data-dir, --model-out, --output-dir and --seed are used by both steps; anything
else (for example --sample-frac) goes to train.py only.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="../../../dataset")
    ap.add_argument("--model-out", default="../model")
    ap.add_argument("--output-dir", default="../../../output")
    args, train_extra = ap.parse_known_args()

    py = sys.executable
    subprocess.run([py, os.path.join(HERE, "train.py"), "--data-dir", args.data_dir,
                    "--model-out", args.model_out, *train_extra], check=True)
    subprocess.run([py, os.path.join(HERE, "predict.py"), "--data-dir", args.data_dir,
                    "--model-dir", args.model_out, "--output-dir", args.output_dir], check=True)


if __name__ == "__main__":
    main()
