"""Teach a PPO network to copy RaceExpert (DAgger), then optionally fine-tune it with PPO.

Why this works better than PPO from scratch:
  * The network gets odor as scale-invariant features (left-right)/(left+right) and log(total)
    via a feature layer inside the policy, so the saved .zip still takes the normal 7 inputs.
  * DAgger: after copying the expert once, the network drives and the expert labels the states
    the network actually visits, so it learns to recover from its own mistakes.

Run from the project root, e.g. for race map A:

  FLYMAZE_MAP_FILE="$PWD/maps/custom_race_a.json" FLYMAZE_TRAIN_MAP=custom_race_a \
  uv run python scripts/distill_ppo.py --output-directory artifacts/race-a-distill

The result is <output>/ppo_flymaze.zip, usable with `flymaze evaluate-ppo` and continue_ppo.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv

from flymaze.evaluation import evaluate_agent
from flymaze.ppo import SB3AgentAdapter, _make_training_env

sys.path.insert(0, str(Path(__file__).resolve().parent))
from race_expert import RaceExpert


class OdorFeatures(BaseFeaturesExtractor):
    """7 raw inputs -> [3 obstacle rays, odor direction, odor strength, speed, turn rate]."""

    def __init__(self, observation_space: gym.spaces.Box) -> None:
        super().__init__(observation_space, features_dim=7)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        odor_left = observations[:, 3:4].clamp(min=0.0)
        odor_right = observations[:, 4:5].clamp(min=0.0)
        total = odor_left + odor_right
        direction = 20.0 * (odor_left - odor_right) / (total + 1e-9)  # ~[-1, 1] in corridors
        strength = (torch.log10(total + 1e-6) + 3.0) / 3.0  # ~[-1, 0.3]
        return torch.cat(
            [observations[:, 0:3], direction.clamp(-3.0, 3.0), strength, observations[:, 5:7]],
            dim=1,
        )


def to_symmetric(action: np.ndarray) -> np.ndarray:
    values = np.asarray(action, dtype=np.float32).copy()
    values[0] = 2.0 * values[0] - 1.0
    return values


def collect(model, vec_env, episodes: int, student_drives: bool, beta: float, rng):
    """Run episodes in parallel; the expert labels every visited state."""
    n_envs = vec_env.num_envs
    experts = [RaceExpert() for _ in range(n_envs)]
    observations = vec_env.reset()
    data_obs, data_act = [], []
    finished, successes = 0, 0
    while finished < episodes:
        expert_actions = np.stack(
            [to_symmetric(experts[i].predict(observations[i])) for i in range(n_envs)]
        )
        data_obs.append(observations.copy())
        data_act.append(expert_actions)
        if student_drives:
            student_actions, _ = model.predict(observations, deterministic=True)
            use_expert = rng.random(n_envs) < beta
            actions = np.where(use_expert[:, None], expert_actions, student_actions)
        else:
            actions = expert_actions
        observations, _, dones, infos = vec_env.step(actions)
        for i, done in enumerate(dones):
            if done:
                experts[i].reset()
                finished += 1
                successes += bool(infos[i].get("success", False))
    return np.concatenate(data_obs), np.concatenate(data_act), successes, finished


def fit(model: PPO, observations, actions, epochs: int, batch_size: int, rng) -> float:
    policy = model.policy
    policy.set_training_mode(True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    obs_t = torch.as_tensor(observations, dtype=torch.float32)
    act_t = torch.as_tensor(np.clip(actions, -1.0, 1.0), dtype=torch.float32)
    loss_value = 0.0
    for _ in range(epochs):
        order = rng.permutation(len(obs_t))
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            mean = policy.get_distribution(obs_t[batch]).distribution.mean
            loss = torch.nn.functional.mse_loss(mean, act_t[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach())
    policy.set_training_mode(False)
    return loss_value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--expert-episodes", type=int, default=16)
    parser.add_argument("--dagger-rounds", type=int, default=3)
    parser.add_argument("--dagger-episodes", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--ppo-timesteps", type=int, default=0, help="0 = skip PPO fine-tuning")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    output = args.output_directory
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    vec_env = make_vec_env(
        _make_training_env,
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "forkserver"},
    )
    log: list[dict[str, object]] = []
    try:
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=1e-4,
            n_steps=512,
            batch_size=128,
            ent_coef=0.0,
            seed=args.seed,
            verbose=1,
            tensorboard_log=str(output / "tensorboard"),
            policy_kwargs={
                "features_extractor_class": OdorFeatures,
                "net_arch": [128, 128],
                "log_std_init": -1.5,
            },
            device="cpu",
        )
        print(f"[distill] expert drives {args.expert_episodes} episodes", flush=True)
        obs, act, ok, n = collect(model, vec_env, args.expert_episodes, False, 1.0, rng)
        log.append({"round": "expert", "success": ok, "episodes": n, "samples": len(obs)})
        loss = fit(model, obs, act, args.epochs, 256, rng)
        print(f"[distill] expert {ok}/{n} success, {len(obs)} samples, loss {loss:.4f}", flush=True)
        for round_index in range(1, args.dagger_rounds + 1):
            beta = 0.5 if round_index == 1 else 0.0
            new_obs, new_act, ok, n = collect(model, vec_env, args.dagger_episodes, True, beta, rng)
            obs, act = np.concatenate([obs, new_obs]), np.concatenate([act, new_act])
            loss = fit(model, obs, act, args.epochs, 256, rng)
            log.append({"round": round_index, "student_success": ok, "episodes": n, "loss": loss})
            print(
                f"[distill] DAgger round {round_index}: network drove {ok}/{n} to food "
                f"(beta={beta}), dataset {len(obs)}, loss {loss:.4f}",
                flush=True,
            )
        model.save(output / "ppo_distilled")
        np.savez_compressed(output / "demonstrations.npz", observations=obs, actions=act)

        if args.ppo_timesteps > 0:
            print(f"[distill] PPO fine-tuning for {args.ppo_timesteps} steps", flush=True)
            checkpoints = CheckpointCallback(
                save_freq=max(8192 // args.n_envs, 1),
                save_path=str(output / "checkpoints"),
                name_prefix="ppo_flymaze",
            )
            model.learn(total_timesteps=args.ppo_timesteps, callback=checkpoints)
        model.save(output / "ppo_flymaze")
    finally:
        vec_env.close()

    adapter = SB3AgentAdapter(PPO.load(output / "ppo_flymaze", device="cpu"))
    seeds = tuple(args.evaluation_seeds)
    report = {
        "model_path": str(output / "ppo_flymaze.zip"),
        "distillation": log,
        "train_evaluation": evaluate_agent("train_l_left", adapter, seeds),
        "held_out_evaluation": evaluate_agent("test_straight_unknown", adapter, seeds),
    }
    (output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    train, held = report["train_evaluation"], report["held_out_evaluation"]
    print(
        f"\nSUMMARY  train map: {train['success_count']}/{train['episode_count']} success"
        f"  |  held-out: {held['success_count']}/{held['episode_count']} success"
    )


if __name__ == "__main__":
    main()