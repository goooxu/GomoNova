"""Training entry point."""

import argparse

from gomonova.training.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="GomoNova training pipeline")
    parser.add_argument("--config", type=str, default="configs/train_main.yaml")
    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
