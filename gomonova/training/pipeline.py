"""Master training pipeline with DDP, parallel MCTS, and opponent diversity.

Three training phases:
  Phase 1 (0 → mcts_start):       Pure policy self-play (fast warmup)
  Phase 2 (mcts_start → renju_start): MCTS-guided, freestyle rules
  Phase 3 (renju_start → end):    MCTS-guided, Renju rules

Self-play runs on ALL CPU cores via ParallelSelfPlay (one worker per core).
Training runs on GPU(s) via DDP.

Launch with: torchrun --nproc_per_node=4 scripts/train.py --config ...
Falls back to single-GPU when launched without torchrun.
"""

from __future__ import annotations

import os
import random
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ..nn.network import GomoNovaNet
from ..utils.checkpoint import load_checkpoint, save_checkpoint
from ..utils.config import load_config
from .evaluator import play_match
from .replay import ReplayBuffer
from .selfplay import generate_games
from .trainer import Trainer


def _setup_ddp() -> tuple[int, int, bool]:
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(rank)
        return rank, world_size, True
    return 0, 1, False


def _cleanup_ddp(is_ddp: bool) -> None:
    if is_ddp:
        dist.destroy_process_group()


def _make_network(model_cfg: dict, device: torch.device) -> GomoNovaNet:
    net = GomoNovaNet(
        channels=model_cfg["channels"],
        num_blocks=model_cfg["num_blocks"],
        policy_channels=model_cfg["policy_channels"],
        value_channels=model_cfg["value_channels"],
    ).bfloat16().to(device)
    for m in net.modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            m.float()
    return net


def run_pipeline(config_path: str) -> None:
    rank, world_size, is_ddp = _setup_ddp()
    device = torch.device(f"cuda:{rank}" if is_ddp else
                          "cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_config(config_path)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    sp_cfg = cfg["selfplay"]
    replay_cfg = cfg["replay"]
    eval_cfg = cfg["eval"]
    phase_cfg = cfg.get("phases", {})

    mcts_start = phase_cfg.get("mcts_start", 500)
    renju_start = phase_cfg.get("renju_start", 2200)
    mcts_sims = sp_cfg.get("mcts_sims", 25)
    mcts_moves = sp_cfg.get("mcts_moves", 10)
    history_ratio = sp_cfg.get("history_opponent_ratio", 0.5)
    history_interval = sp_cfg.get("history_save_interval", 200)
    history_keep = sp_cfg.get("history_keep", 10)

    # --- Networks ---
    network = _make_network(model_cfg, device)
    best_network = _make_network(model_cfg, device)

    if rank == 0:
        print(f"Device: {device}, GPUs: {world_size}, DDP: {is_ddp}")
        print(f"Model params: {network.num_params():,} (BF16 weights, BN=FP32)")

    ckpt_dir = cfg.get("checkpoint_dir", "checkpoints")
    if rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "best.pt")

    start_iter = 0
    if os.path.exists(best_path):
        start_iter = load_checkpoint(best_path, network, device)
        load_checkpoint(best_path, best_network, device)
        if rank == 0:
            print(f"Resumed from iteration {start_iter}")

    if is_ddp:
        network = DDP(network, device_ids=[rank])

    def _raw_model(net):
        return net.module if isinstance(net, DDP) else net

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

    history_pool: list[str] = []
    total_iters = train_cfg["total_iters"]
    games_per_iter = sp_cfg["games_per_iter"]

    for iteration in range(start_iter, total_iters):
        t0 = time.time()
        lr = trainer.update_lr(iteration)

        use_mcts = iteration >= mcts_start
        use_renju = iteration >= renju_start

        # --- Self-play ---
        _raw_model(network).eval()
        samples = generate_games(
            _raw_model(network), device,
            num_games=games_per_iter,
            temp_threshold=sp_cfg.get("temp_threshold", 20),
            augment=4,
            use_mcts=use_mcts,
            mcts_sims=mcts_sims,
            mcts_moves=mcts_moves,
            use_renju=use_renju,
            dirichlet_alpha=sp_cfg.get("dirichlet_alpha", 0.3),
        )
        replay.add_batch(samples)
        sp_time = time.time() - t0

        if len(replay) < train_cfg["batch_size"] * 2:
            if rank == 0:
                print(f"Iter {iteration}: collecting data ({len(replay)} samples)")
            continue

        # --- Train (GPU) ---
        network.train()
        t1 = time.time()
        metrics = trainer.train_epoch(
            replay, train_cfg["batch_size"], train_cfg["train_steps_per_iter"]
        )
        train_time = time.time() - t1

        # --- Evaluate (rank 0 only) ---
        eval_time = 0.0
        promoted = False
        winrate = -1.0
        if rank == 0:
            network.eval()
            t2 = time.time()
            result = play_match(
                _raw_model(network), best_network, device,
                num_games=eval_cfg["games"],
            )
            eval_time = time.time() - t2
            winrate = result["winrate"]

            promoted = winrate >= eval_cfg["promote_winrate"]
            if promoted:
                best_network.load_state_dict(_raw_model(network).state_dict())
                save_checkpoint(
                    best_path, _raw_model(network),
                    trainer.optimizer, iteration + 1,
                )

        # --- Save historical checkpoint ---
        if rank == 0 and (iteration + 1) % history_interval == 0:
            hist_path = os.path.join(ckpt_dir, f"model_{iteration+1:04d}.pt")
            save_checkpoint(
                hist_path, _raw_model(network),
                trainer.optimizer, iteration + 1,
            )
            history_pool.append(hist_path)
            if len(history_pool) > history_keep:
                old = history_pool.pop(0)
                if os.path.exists(old):
                    os.remove(old)

        if rank == 0:
            phase = "MCTS+Renju" if use_renju else "MCTS" if use_mcts else "Policy"
            print(
                f"Iter {iteration+1}/{total_iters} | "
                f"LR {lr:.5f} | "
                f"Loss {metrics['loss']:.4f} "
                f"(P:{metrics['policy_loss']:.3f} V:{metrics['value_loss']:.3f}) | "
                f"Eval WR {winrate:.2%} {'*' if promoted else ''} | "
                f"SP {sp_time:.0f}s Train {train_time:.0f}s Eval {eval_time:.0f}s | "
                f"Buf {len(replay)} | {phase}"
            )

    # --- Final save ---
    if rank == 0:
        save_checkpoint(best_path, best_network, trainer.optimizer, total_iters)
        print("Training complete.")

    _cleanup_ddp(is_ddp)
