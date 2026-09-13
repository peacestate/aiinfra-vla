"""ACT baseline policy — the non-language-conditioned floor.

⚠️ The published checkpoint `lerobot/act_aloha_sim_transfer_cube_human` predates
LeRobot 0.4.x. Loading it emits only a *warning* about "unexpected keys"
(`normalize_inputs.*`, `unnormalize_outputs.*`) and hands back a policy with NO
normalization at all — in 0.4.x normalization lives in a separate processor
pipeline, not in the policy module.

Nothing crashes. The policy runs, returns finite-looking actions, and fails the
task. That reads as "ACT is a weak baseline" when the truth is "the stats never
loaded". So we rebuild the processors explicitly from the dataset statistics.
"""
import logging

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors

CHECKPOINT = "lerobot/act_aloha_sim_transfer_cube_human"
DATASET = "lerobot/aloha_sim_transfer_cube_human"


class ACTBaseline:
    """ACT is *not* language-conditioned; `instruction` is accepted and ignored.
    That is precisely why it is only the floor and not the submission."""

    name = "act_pytorch_cpu"

    def __init__(self, checkpoint=CHECKPOINT, dataset=DATASET, device="cpu"):
        self.device = torch.device(device)
        self.policy = ACTPolicy.from_pretrained(checkpoint).to(self.device).eval()

        stats = LeRobotDatasetMetadata(dataset).stats
        self.pre, self.post = make_act_pre_post_processors(
            self.policy.config, dataset_stats=stats
        )
        self._assert_normalized(stats)

    def _assert_normalized(self, stats):
        """Fail loudly here rather than silently scoring 0% later."""
        if not stats:
            raise RuntimeError(f"no dataset stats for {DATASET}")
        for key in ("observation.state", "action"):
            if key not in stats:
                raise RuntimeError(f"dataset stats missing '{key}': got {list(stats)}")
            mean = torch.as_tensor(stats[key]["mean"])
            std = torch.as_tensor(stats[key]["std"])
            if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
                raise RuntimeError(f"non-finite stats for '{key}'")
            if (std == 0).any():
                raise RuntimeError(f"zero-variance stats for '{key}' — would divide by 0")
        logging.info("ACT normalization stats verified for %s", list(stats))

    def reset(self):
        self.policy.reset()
        self.did_forward = False

    def act(self, obs, instruction=None):
        # ACT chunks: it runs the network only when the action queue is empty and
        # replays cached actions otherwise. Timing every call together averages a
        # ~1000 ms forward pass into ~200 cheap queue pops and reports ~10 ms,
        # which is not the inference latency of anything.
        self.did_forward = len(self.policy._action_queue) == 0
        batch = {
            "observation.state": torch.from_numpy(
                np.asarray(obs["agent_pos"], dtype=np.float32)
            ),
            "observation.images.top": torch.from_numpy(
                np.asarray(obs["pixels"]["top"], dtype=np.float32) / 255.0
            ).permute(2, 0, 1),
        }
        with torch.no_grad():
            processed = self.pre(batch)
            action = self.policy.select_action(processed)
            action = self.post(action)
        return action.squeeze(0).float().cpu().numpy()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import gymnasium as gym
    import gym_aloha  # noqa: F401

    p = ACTBaseline()
    env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos")
    obs, _ = env.reset(seed=0)
    p.reset()
    a = p.act(obs)
    print("action:", a.shape, "finite:", np.isfinite(a).all())
    print(f"  range [{a.min():.4f}, {a.max():.4f}]")
    print(f"  start state range [{obs['agent_pos'].min():.4f}, {obs['agent_pos'].max():.4f}]")
    print("  delta from current state:", np.abs(a - obs["agent_pos"]).max())
    env.close()
