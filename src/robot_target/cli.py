"""Each GPU command runs independently for auditable loading and memory use."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["download", "prepare", "evaluate", "train", "verify-reload"])
    parser.add_argument("--config", default="configs/smoke.json")
    parser.add_argument("--data", default="/content")
    parser.add_argument("--manifests", default="artifacts/manifests")
    parser.add_argument("--output", default="artifacts/smoke")
    parser.add_argument("--adapter")
    parser.add_argument("--splits", nargs="+", default=["val"])
    parser.add_argument("--variants", nargs="+", default=["original"])
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.command == "download":
        from .data import download
        download(args.data)
    elif args.command == "prepare":
        from .data import prepare
        print(json.dumps(prepare(args.data, args.manifests, config), indent=2))
    else:
        from .experiment import run_evaluation, train, verify_reload
        if args.command == "evaluate":
            run_evaluation(config, args.manifests, args.data, args.output, args.adapter, args.splits, args.variants)
        elif args.command == "train":
            train(config, args.manifests, args.data, args.output)
        else:
            verify_reload(config, args.manifests, args.data, args.output)


if __name__ == "__main__":
    main()
