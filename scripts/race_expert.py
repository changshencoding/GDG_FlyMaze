"""RaceExpert: a rule-based teacher for FlyMaze race maps (uses only the 7 standard inputs).

Ideas that make it work on long corridors:
  * Odor steering is scale-invariant: (left - right) / (left + right). Far from the food the raw
    odor is ~1e-4, so a fixed gain on the raw difference (as in StatelessRuleAgent) does nothing.
  * When a wall is ahead it commits to one turn direction until the way is clear (no dithering).
  * If odor drops well below the best value seen, it assumes it passed the food and turns around.
  * Once the food is a few mm away (strong odor) it walks straight at it and ignores walls,
    so it does not circle around food that sits in a corner; if a wall touches its nose there,
    the food must be beside or behind it, so it turns in place instead of pushing the wall.

Evaluate it, e.g. on race map A:

  FLYMAZE_MAP_FILE="$PWD/maps/custom_race_a.json" \
  uv run python scripts/race_expert.py --map-id custom_race_a --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from numpy.typing import NDArray


class RaceExpert:
    def __init__(
        self,
        k_odor: float = 60.0,
        wall_turn: float = 1.0,
        wall_forward: float = 0.4,
        side_gain: float = 0.8,
        side_threshold: float = 0.8,
        forward: float = 1.0,
        odor_threshold: float = 0.003,
        close_front: float = 0.4,
        uturn_drop: float = 0.6,
        uturn_steps: int = 12,
        near_food: float = 0.9,
        near_forward: float = 0.4,
        nose_forward: float = 0.1,
    ) -> None:
        self.k_odor = k_odor
        self.wall_turn = wall_turn
        self.wall_forward = wall_forward
        self.side_gain = side_gain
        self.side_threshold = side_threshold
        self.forward = forward
        self.odor_threshold = odor_threshold
        self.close_front = close_front
        self.uturn_drop = uturn_drop
        self.uturn_steps = uturn_steps
        self.near_forward = near_forward  # walking speed once the food is within a few mm
        self.nose_forward = nose_forward  # drive used to turn while the nose touches a wall
        self.near_food = near_food  # odor total above this = food is right here (0 = off)
        self.reset()

    def reset(self) -> None:
        self.commit = 0.0  # turn direction held while a wall is ahead
        self.max_odor = 0.0  # strongest total odor seen this episode
        self.uturn_left = 0  # remaining steps of a U-turn after overshooting the food
        self.uturn_direction = 1.0

    def predict(self, observation: NDArray[np.float32]) -> NDArray[np.float32]:
        front, front_left, front_right, odor_left, odor_right, _, _ = np.asarray(
            observation, dtype=np.float64
        )
        total = odor_left + odor_right
        odor = (odor_left - odor_right) / total if total > 1e-12 else 0.0  # > 0: food on the left

        if self.uturn_left == 0 and self.max_odor > 0.3 and total < self.uturn_drop * self.max_odor:
            self.uturn_left = self.uturn_steps
            self.uturn_direction = 1.0 if odor >= 0 else -1.0
            self.max_odor = total
        self.max_odor = max(self.max_odor, total)
        if self.near_food > 0 and total > self.near_food:
            # food is within a few mm: steer straight at it, walls are not a reason to turn away...
            self.uturn_left = 0
            if front < 0.3:
                # ...unless a wall touches our nose: the food is beside/behind us, turn away
                if self.commit == 0.0:
                    if abs(odor) > self.odor_threshold:
                        self.commit = 1.0 if odor > 0 else -1.0
                    else:
                        self.commit = 1.0 if front_left >= front_right else -1.0
                return np.asarray([self.nose_forward, self.commit], dtype=np.float32)
            self.commit = 0.0
            turn = float(np.clip(self.k_odor * odor, -1.0, 1.0))
            return np.asarray([self.near_forward, turn], dtype=np.float32)
        if self.uturn_left > 0:
            self.uturn_left -= 1
            return np.asarray([0.2, self.uturn_direction], dtype=np.float32)

        turn = self.k_odor * odor
        if front_left < self.side_threshold:
            turn -= self.side_gain * (self.side_threshold - front_left) / self.side_threshold
        if front_right < self.side_threshold:
            turn += self.side_gain * (self.side_threshold - front_right) / self.side_threshold
        forward = self.forward
        if front < 1.0:
            if self.commit == 0.0:
                if abs(odor) > self.odor_threshold:
                    self.commit = 1.0 if odor > 0 else -1.0
                else:
                    self.commit = 1.0 if front_left > front_right else -1.0
            turn = self.commit * self.wall_turn
            forward = self.wall_forward if front > self.close_front else 0.1
        else:
            self.commit = 0.0
        turn = float(np.clip(turn, -1.0, 1.0))
        return np.asarray([forward * (1.0 - 0.3 * abs(turn)), turn], dtype=np.float32)


def main() -> None:
    from flymaze.evaluation import evaluate_agent

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-id", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--near-food", type=float, default=0.9)
    args = parser.parse_args()
    report = evaluate_agent(args.map_id, RaceExpert(near_food=args.near_food), tuple(args.seeds))
    for result in report["results"]:  # type: ignore[union-attr]
        print(
            f"seed {result['seed']}: success={result['success']} steps={result['steps']} "
            f"fallen={result['fallen']} wall_steps={result['wall_contact_steps']} "
            f"remaining={result['final_route_distance_mm']:.1f} mm"
        )
    print(json.dumps({k: v for k, v in report.items() if k != "results"}))
    print(f"\nSUMMARY  {args.map_id}: {report['success_count']}/{report['episode_count']} success")


if __name__ == "__main__":
    main()