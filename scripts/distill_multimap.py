"""Train ONE network on several maps at once by copying RaceExpert (DAgger).

Each parallel environment runs a different map (round robin), so every round sees every map.
Run from the project root, e.g.

  uv run python scripts/distill_multimap.py --output-directory artifacts/multimap-v1 \
    --maps maps/custom_race_a.json maps/custom_raceb.json maps/custom_left.json \
           maps/custom_right.json maps/custom_s_turn.json

Options:
  --train-timeout 30   give maps that are too long to finish (e.g. race B) more time
                       DURING TRAINING only;
                       evaluation always uses each map's official timeout.
  --init-model PATH    continue from an earlier model
                       (e.g. artifacts/race-a-distill/ppo_flymaze.zip)
  --init-data PATH     reuse earlier demonstrations
                       (e.g. artifacts/race-a-distill/demonstrations.npz)

The result <output>/ppo_flymaze.zip takes the normal 7 inputs and works with
`flymaze evaluate-ppo` (pass FLYMAZE_MAP_FILE for custom maps as usual).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv

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
        direction = 20.0 * (odor_left - odor_right) / (total + 1e-9)
        strength = (torch.log10(total + 1e-6) + 3.0) / 3.0
        return torch.cat(
            [observations[:, 0:3], direction.clamp(-3.0, 3.0), strength, observations[:, 5:7]],
            dim=1,
        )


def register_map(path: str, timeout: float | None = None) -> str:
    """Load a custom JSON map into flymaze's registry and return its map_id."""
    from flymaze.custom_map_runtime import make_map_definition
    from flymaze.custom_maps import read_grid_map
    from flymaze.maps import MAPS

    definition = make_map_definition(read_grid_map(Path(path).resolve(strict=True)))
    # only extend maps that are too long to finish in time (route / 12 mm/s > 90% of timeout)
    too_long = definition.route.total_length_mm / 12.0 > 0.9 * definition.timeout_seconds
    if timeout is not None and too_long and timeout > definition.timeout_seconds:
        definition = dataclasses.replace(definition, timeout_seconds=timeout)
    MAPS[definition.map_id] = definition
    return definition.map_id


def make_env(path: str, timeout: float | None) -> gym.Env:
    from flymaze.env import FlyMazeEnv
    from flymaze.ppo import SymmetricPPOActionWrapper

    return SymmetricPPOActionWrapper(FlyMazeEnv(register_map(path, timeout)))


def to_symmetric(action: np.ndarray) -> np.ndarray:
    values = np.asarray(action, dtype=np.float32).copy()
    values[0] = 2.0 * values[0] - 1.0
    return values


def collect(model, vec_env, env_names, episodes, student_drives, beta, rng):
    n_envs = vec_env.num_envs
    experts = [RaceExpert() for _ in range(n_envs)]
    observations = vec_env.reset()
    data_obs, data_act = [], []
    tally: dict[str, list[int]] = {name: [0, 0] for name in env_names}
    finished = np.zeros(n_envs, dtype=int)
    while finished.min() < episodes:  # every env (so every map) finishes >= `episodes`
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
                finished[i] += 1
                tally[env_names[i]][0] += bool(infos[i].get("success", False))
                tally[env_names[i]][1] += 1
    text = ", ".join(f"{name} {ok}/{n}" for name, (ok, n) in tally.items())
    return np.concatenate(data_obs), np.concatenate(data_act), text


def fit(model: PPO, observations, actions, epochs: int, rng) -> float:
    policy = model.policy
    policy.set_training_mode(True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    obs_t = torch.as_tensor(observations, dtype=torch.float32)
    act_t = torch.as_tensor(np.clip(actions, -1.0, 1.0), dtype=torch.float32)
    loss_value = 0.0
    for _ in range(epochs):
        order = rng.permutation(len(obs_t))
        for start in range(0, len(order), 256):
            batch = order[start : start + 256]
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
    parser.add_argument("--maps", nargs="+", required=True)
    parser.add_argument("--n-envs", type=int, default=10)
    parser.add_argument("--train-timeout", type=float, default=30.0)
    parser.add_argument("--expert-episodes", type=int, default=2, help="per environment")
    parser.add_argument("--dagger-rounds", type=int, default=4)
    parser.add_argument("--dagger-episodes", type=int, default=2, help="per environment")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--init-model", type=Path)
    parser.add_argument("--init-data", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    output = args.output_directory
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    map_paths = [str(Path(p).resolve(strict=True)) for p in args.maps]
    map_ids = [register_map(p) for p in map_paths]
    env_paths = [map_paths[i % len(map_paths)] for i in range(args.n_envs)]
    env_names = [map_ids[i % len(map_ids)] for i in range(args.n_envs)]
    print(f"[multimap] training maps: {', '.join(map_ids)}", flush=True)

    vec_env = SubprocVecEnv(
        [partial(make_env, path, args.train_timeout) for path in env_paths],
        start_method="forkserver",
    )
    vec_env.seed(args.seed)
    log: list[dict[str, object]] = []
    try:
        if args.init_model:
            model = PPO.load(args.init_model, env=vec_env, device="cpu")
            print(f"[multimap] continuing from {args.init_model}", flush=True)
        else:
            model = PPO(
                "MlpPolicy",
                vec_env,
                learning_rate=1e-4,
                n_steps=512,
                batch_size=128,
                ent_coef=0.0,
                seed=args.seed,
                verbose=0,
                policy_kwargs={
                    "features_extractor_class": OdorFeatures,
                    "net_arch": [128, 128],
                    "log_std_init": -1.5,
                },
                device="cpu",
            )
        print(f"[multimap] expert drives {args.expert_episodes} episodes per env", flush=True)
        obs, act, text = collect(model, vec_env, env_names, args.expert_episodes, False, 1.0, rng)
        if args.init_data:
            old = np.load(args.init_data)
            obs = np.concatenate([old["observations"], obs])
            act = np.concatenate([old["actions"], act])
        loss = fit(model, obs, act, args.epochs, rng)
        log.append({"round": "expert", "result": text, "samples": len(obs), "loss": loss})
        print(f"[multimap] expert: {text} | samples {len(obs)} | loss {loss:.4f}", flush=True)
        for round_index in range(1, args.dagger_rounds + 1):
            beta = 0.5 if round_index == 1 else 0.0
            new_obs, new_act, text = collect(
                model, vec_env, env_names, args.dagger_episodes, True, beta, rng
            )
            obs, act = np.concatenate([obs, new_obs]), np.concatenate([act, new_act])
            loss = fit(model, obs, act, args.epochs, rng)
            log.append({"round": round_index, "beta": beta, "result": text, "loss": loss})
            print(
                f"[multimap] round {round_index} (network drives, beta={beta}): {text}"
                f" | samples {len(obs)} | loss {loss:.4f}",
                flush=True,
            )
            model.save(output / f"ppo_round{round_index}")
            np.savez_compressed(output / "demonstrations.npz", observations=obs, actions=act)
        model.save(output / "ppo_flymaze")
    finally:
        vec_env.close()

    from flymaze.evaluation import evaluate_agent
    from flymaze.maps import MAPS
    from flymaze.ppo import SB3AgentAdapter

    for path in map_paths:  # official timeouts for evaluation
        register_map(path)
    adapter = SB3AgentAdapter(PPO.load(output / "ppo_flymaze", device="cpu"))
    seeds = tuple(args.evaluation_seeds)
    unseen = [m for m in ("test_straight_unknown", "validation_s_turn_complex") if m in MAPS]
    evaluations = {}
    for map_id in [*map_ids, *unseen]:
        evaluations[map_id] = evaluate_agent(map_id, adapter, seeds)
        result = evaluations[map_id]
        tag = "(never trained on)" if map_id in unseen else ""
        print(
            f"[multimap] eval {map_id}: {result['success_count']}/{result['episode_count']} {tag}",
            flush=True,
        )
    report = {
        "model_path": str(output / "ppo_flymaze.zip"),
        "rounds": log,
        "evaluation": evaluations,
    }
    (output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    summary = "  ".join(
        f"{m}: {r['success_count']}/{r['episode_count']}" for m, r in evaluations.items()
    )
    print(f"\nSUMMARY  {summary}")


if __name__ == "__main__":
    main()