"""Master training pipeline: selfplay -> train -> evaluate -> promote."""

from __future__ import annotations

import os
import time

import torch

from ..nn.network import GomoNovaNet
from ..utils.checkpoint import load_checkpoint, save_checkpoint
from ..utils.config import load_config
from .evaluator import play_match
from .replay import ReplayBuffer
from .selfplay import generate_games
from .trainer import Trainer


def run_pipeline(config_path: str) -> None:
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Device: {device}, GPUs: {num_gpus}")

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    sp_cfg = cfg["selfplay"]
    replay_cfg = cfg["replay"]
    eval_cfg = cfg["eval"]

    network = GomoNovaNet(
        channels=model_cfg["channels"],
        num_blocks=model_cfg["num_blocks"],
        policy_channels=model_cfg["policy_channels"],
        value_channels=model_cfg["value_channels"],
    ).bfloat16().to(device)
    for m in network.modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            m.float()
    print(f"Model params: {network.num_params():,} (BF16 weights, BN=FP32)")

    best_network = GomoNovaNet(
        channels=model_cfg["channels"],
        num_blocks=model_cfg["num_blocks"],
        policy_channels=model_cfg["policy_channels"],
        value_channels=model_cfg["value_channels"],
    ).bfloat16().to(device)
    for m in best_network.modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            m.float()

    ckpt_dir = cfg.get("checkpoint_dir", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "best.pt")

    start_iter = 0
    if os.path.exists(best_path):
        start_iter = load_checkpoint(best_path, network, device)
        load_checkpoint(best_path, best_network, device)
        print(f"Resumed from iteration {start_iter}")

    # Wrap in DataParallel after loading checkpoint
    # Note: DataParallel is incompatible with BF16 master weights.
    # Single GPU is sufficient for this model size (12.5M params).

    def _raw_model(net):
        return net.module if isinstance(net, torch.nn.DataParallel) else net

    best_network.load_state_dict(_raw_model(network).state_dict())

    trainer = Trainer(
        network, device,
        lr=train_cfg["lr"],
        lr_min=train_cfg["lr_min"],
        weight_decay=train_cfg["weight_decay"],
        warmup_iters=train_cfg["warmup_iters"],
        total_iters=train_cfg["total_iters"],
        grad_clip=train_cfg["grad_clip"],
        use_amp=train_cfg.get("use_amp", True),
    )
    replay = ReplayBuffer(capacity=replay_cfg["capacity"])

    total_iters = train_cfg["total_iters"]
    for iteration in range(start_iter, total_iters):
        t0 = time.time()
        lr = trainer.update_lr(iteration)

        network.eval()
        samples = generate_games(
            network, device,
            num_games=sp_cfg["games_per_iter"],
            temp_threshold=sp_cfg.get("temp_threshold", 30),
            augment=4,
        )
        replay.add_batch(samples)
        sp_time = time.time() - t0

        if len(replay) < train_cfg["batch_size"] * 2:
            print(f"Iter {iteration}: collecting more data ({len(replay)} samples)")
            continue

        network.train()
        t1 = time.time()
        metrics = trainer.train_epoch(
            replay, train_cfg["batch_size"], train_cfg["train_steps_per_iter"]
        )
        train_time = time.time() - t1

        network.eval()
        t2 = time.time()
        result = play_match(network, best_network, device, num_games=eval_cfg["games"])
        eval_time = time.time() - t2

        promoted = result["winrate"] >= eval_cfg["promote_winrate"]
        if promoted:
            best_network.load_state_dict(_raw_model(network).state_dict())
            save_checkpoint(best_path, _raw_model(network), trainer.optimizer, iteration + 1)

        if (iteration + 1) % 100 == 0:
            path = os.path.join(ckpt_dir, f"model_{iteration+1:04d}.pt")
            save_checkpoint(path, _raw_model(network), trainer.optimizer, iteration + 1)

        print(
            f"Iter {iteration+1}/{total_iters} | "
            f"LR {lr:.5f} | "
            f"Loss {metrics['loss']:.4f} (P:{metrics['policy_loss']:.3f} V:{metrics['value_loss']:.3f}) | "
            f"Eval WR {result['winrate']:.2%} {'*' if promoted else ''} | "
            f"SP {sp_time:.0f}s Train {train_time:.0f}s Eval {eval_time:.0f}s | "
            f"Buffer {len(replay)}"
        )

    save_checkpoint(best_path, best_network, trainer.optimizer, total_iters)
    print("Training complete.")
