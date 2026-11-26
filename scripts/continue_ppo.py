"""Continue training an existing FlyMaze PPO model, with periodic checkpoints.

Run from the project root with the same map variables used for training, e.g.

  FLYMAZE_MAP_FILE="$PWD/maps/custom_race_a.json" FLYMAZE_TRAIN_MAP=custom_race_a \
  uv run python scripts/continue_ppo.py \
    --model-path artifacts/race-a-v4/ppo_flymaze.zip \
    --output-directory artifacts/race-a-v5 --timesteps 50000

A checkpoint is saved every --checkpoint-every steps into <output>/checkpoints,
so an interrupted run can be resumed by passing the newest checkpoint as --model-path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from flymaze.evaluation import evaluate_agent
from flymaze.ppo import SB3AgentAdapter, _make_training_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=8192, help="total env steps")
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    output = args.output_directory
    output.mkdir(parents=True, exist_ok=True)
    vec_env = make_vec_env(
        _make_training_env,
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "forkserver"},
    )
    model_path = output / "ppo_flymaze"
    try:
        model = PPO.load(
            args.model_path,
            env=vec_env,
            device="cpu",
            tensorboard_log=str(output / "tensorboard"),
        )
        print(f"Loaded {args.model_path} (already trained {model.num_timesteps} steps)")
        checkpoints = CheckpointCallback(
            save_freq=max(args.checkpoint_every // args.n_envs, 1),
            save_path=str(output / "checkpoints"),
            name_prefix="ppo_flymaze",
        )
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoints,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        model.save(model_path)
        total_steps = model.num_timesteps
    finally:
        vec_env.close()

    adapter = SB3AgentAdapter(PPO.load(model_path, device="cpu"))
    report = {
        "base_model": str(args.model_path),
        "model_path": str(model_path.with_suffix(".zip")),
        "total_trained_steps": total_steps,
        "train_evaluation": evaluate_agent("train_l_left", adapter, tuple(args.evaluation_seeds)),
        "held_out_evaluation": evaluate_agent(
            "test_straight_unknown", adapter, tuple(args.evaluation_seeds)
        ),
    }
    (output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    train, held = report["train_evaluation"], report["held_out_evaluation"]
    print(json.dumps(report, indent=2))
    print(
        f"\nSUMMARY  train map: {train['success_count']}/{train['episode_count']} success"
        f"  |  held-out: {held['success_count']}/{held['episode_count']} success"
    )


if __name__ == "__main__":
    main()