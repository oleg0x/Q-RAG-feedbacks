"""Compose Q-RAG Hydra configs without instantiating models or datasets."""

import argparse
from pathlib import Path

from hydra import compose, initialize
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="*", help="Config names without .yaml")
    parser.add_argument("--all", action="store_true", help="Compose all training/testing configs")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--show", action="store_true", help="Print resolved YAML")
    args = parser.parse_args()

    names = list(args.configs)
    if args.all:
        names.extend(
            sorted(
                path.stem
                for path in Path("configs").glob("*.yaml")
                if path.name.startswith(("training", "testing"))
            )
        )
    names = list(dict.fromkeys(names))
    if not names:
        parser.error("provide config names or --all")

    with initialize(version_base="1.3", config_path="../configs"):
        for name in names:
            cfg = compose(config_name=name, overrides=args.override)
            OmegaConf.to_container(cfg, resolve=True)
            print(f"OK {name}")
            if args.show:
                print(OmegaConf.to_yaml(cfg, resolve=True))


if __name__ == "__main__":
    main()
