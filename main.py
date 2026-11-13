from __future__ import annotations

import argparse
from importlib import import_module

from utils.config import CONFIG_ROOT


TASK_REGISTRY: dict[str, dict[str, object]] = {
    "train": {
        "module": "tasks.train",
        "handler_name": "run",
        "default_config": CONFIG_ROOT / "tcs_seg" / "train" / "deepcrack.yaml",
        "help": "Run model training.",
    },
    "infer": {
        "module": "tasks.infer",
        "handler_name": "run",
        "default_config": CONFIG_ROOT / "tcs_seg" / "infer" / "deepcrack.yaml",
        "help": "Run single-image inference.",
    },
    "eval": {
        "module": "tasks.eval",
        "handler_name": "run",
        "default_config": CONFIG_ROOT / "tcs_seg" / "eval" / "deepcrack.yaml",
        "help": "Run evaluation metrics.",
    },
    "flops": {
        "module": "tasks.flops",
        "handler_name": "run",
        "default_config": CONFIG_ROOT / "tcs_seg" / "flops.yaml",
        "help": "Calculate FLOPs and parameter count.",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TCS-Seg unified experiment entrypoint.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command_name, metadata in TASK_REGISTRY.items():
        subparser = subparsers.add_parser(command_name, help=metadata["help"])
        subparser.add_argument(
            "--config",
            type=str,
            default=str(metadata["default_config"]),
            help="Path to the YAML config file.",
        )
        if command_name == "eval":
            subparser.add_argument(
                "--checkpoint",
                type=str,
                default=None,
                help="Optional eval checkpoint override: best, latest, or a checkpoint path.",
            )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    command = TASK_REGISTRY[args.command]
    module = import_module(command["module"])
    handler = getattr(module, command["handler_name"])
    if args.command == "eval":
        handler(args.config, checkpoint_override=args.checkpoint)
    else:
        handler(args.config)


if __name__ == "__main__":
    main()
