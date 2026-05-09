from __future__ import annotations

import argparse

from src.data.eda import run_all as run_eda
from src.data.features import build_all
from src.evaluation.report import run as run_report
from src.models import classification, clustering, ensemble, regression
from src.rl import train as rl_train
from src.utils.seed import set_seed


def main(skip_features: bool = False, skip_classification: bool = False, skip_rl: bool = False):
    set_seed()

    if not skip_features:
        print(">> building features")
        build_all()
        run_eda()

    if not skip_classification:
        print(">> classification")
        classification.run()

    print(">> regression")
    regression.run()

    print(">> clustering")
    clustering.run()

    print(">> ensemble")
    ensemble.run()

    if not skip_rl:
        print(">> reinforcement learning")
        rl_train.main()

    print(">> report")
    run_report()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--skip-features", action="store_true")
    p.add_argument("--skip-classification", action="store_true")
    p.add_argument("--skip-rl", action="store_true")
    args = p.parse_args()
    main(
        skip_features=args.skip_features,
        skip_classification=args.skip_classification,
        skip_rl=args.skip_rl,
    )
