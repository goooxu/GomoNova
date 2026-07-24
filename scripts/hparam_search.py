"""Hyperparameter search: run short training experiments and compare."""
import os
import sys
import yaml
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_CFG = {
    "board_size": 15,
    "game_rules": "renju",
    "model": {"channels": 192, "num_blocks": 10, "policy_channels": 96, "value_channels": 48},
    "training": {
        "batch_size": 2048, "lr": 1e-4, "lr_min": 1e-6,
        "warmup_iters": 5, "total_iters": 50,
        "train_steps_per_iter": 64, "weight_decay": 1e-4,
        "grad_clip": 1.0, "use_amp": True,
    },
    "selfplay": {
        "games_per_iter": 512, "temp_threshold": 20,
        "dirichlet_alpha": 0.3,
        "history_opponent_ratio": 0.0,
        "history_save_interval": 999, "history_keep": 2,
    },
    "replay": {"capacity": 500000},
    "eval": {"games": 50, "promote_winrate": 0.55},
}

EXPERIMENTS = {
    "A_10s3m":  {"mcts_sims": 10, "mcts_moves": 3},
    "B_25s5m":  {"mcts_sims": 25, "mcts_moves": 5},
    "C_50s10m": {"mcts_sims": 50, "mcts_moves": 10},
    "D_25s10m": {"mcts_sims": 25, "mcts_moves": 10},
}


def run_experiment(name: str, overrides: dict) -> None:
    import copy
    cfg = copy.deepcopy(BASE_CFG)
    cfg["selfplay"]["mcts_sims"] = overrides["mcts_sims"]
    cfg["selfplay"]["mcts_moves"] = overrides["mcts_moves"]
    cfg["phases"] = {"mcts_start": 5, "renju_start": 999}
    cfg["checkpoint_dir"] = f"ckpt_{name}"

    path = f"/tmp/exp_{name}.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f)

    if os.path.exists(cfg["checkpoint_dir"]):
        shutil.rmtree(cfg["checkpoint_dir"])

    print(f"\n{'='*60}")
    print(f"  EXP {name}: sims={overrides['mcts_sims']}, moves={overrides['mcts_moves']}")
    print(f"{'='*60}", flush=True)

    from gomonova.training.pipeline import run_pipeline
    run_pipeline(path)


if __name__ == "__main__":
    selected = sys.argv[1:] if len(sys.argv) > 1 else list(EXPERIMENTS.keys())
    for name in selected:
        if name in EXPERIMENTS:
            run_experiment(name, EXPERIMENTS[name])
