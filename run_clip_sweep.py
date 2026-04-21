import argparse
import json
import os
import subprocess
import sys


def load_registry(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config.get("clip_models", {})


def main():
    parser = argparse.ArgumentParser(description="Run EEG_retrieval.py across multiple local CLIP models.")
    parser.add_argument(
        "--config",
        type=str,
        default="data_config.json",
        help="Path to data_config.json",
    )
    parser.add_argument(
        "--clip_variants",
        nargs="+",
        default=["all"],
        help='CLIP aliases to run, or "all" to use every configured model.',
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List configured CLIP aliases and exit.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the commands without executing them.",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue running remaining CLIP models if one run fails.",
    )
    args, forward_args = parser.parse_known_args()
    if forward_args and forward_args[0] == "--":
        forward_args = forward_args[1:]

    registry = load_registry(args.config)
    if not registry:
        raise RuntimeError(f"No clip_models found in {args.config}")

    if args.list:
        for alias, model_cfg in registry.items():
            print(f"{alias}: model={model_cfg.get('model_name')} path={model_cfg.get('pretrained_path', '')}")
        return

    if len(args.clip_variants) == 1 and args.clip_variants[0] == "all":
        selected = list(registry.keys())
    else:
        unknown = [alias for alias in args.clip_variants if alias not in registry]
        if unknown:
            raise ValueError(f"Unknown clip variants: {unknown}. Available: {sorted(registry.keys())}")
        selected = args.clip_variants

    for alias in selected:
        model_cfg = registry[alias]
        pretrained_path = model_cfg.get("pretrained_path", "")
        if not pretrained_path:
            message = f"[skip] {alias}: pretrained_path is empty in {args.config}"
            if args.continue_on_error:
                print(message)
                continue
            raise ValueError(message)
        if not os.path.exists(pretrained_path):
            message = f"[skip] {alias}: checkpoint not found at {pretrained_path}"
            if args.continue_on_error:
                print(message)
                continue
            raise FileNotFoundError(message)

        env = os.environ.copy()
        env["CLIP_VARIANT"] = alias
        env["CLIP_MODEL_NAME"] = model_cfg["model_name"]
        env["CLIP_PRETRAINED_PATH"] = pretrained_path

        cmd = [sys.executable, "EEG_retrieval.py", *forward_args]
        print(f"[run] {alias}: {' '.join(cmd)}")

        if args.dry_run:
            continue

        completed = subprocess.run(cmd, env=env)
        if completed.returncode != 0:
            if args.continue_on_error:
                print(f"[warn] {alias} failed with exit code {completed.returncode}")
                continue
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
