from collections import defaultdict
from typing import Callable
import numpy as np
import torch
from tqdm.auto import tqdm
from mani_skill.utils.gym_utils import find_max_episode_steps_value

def evaluate(n: int, sample_fn: Callable, eval_envs):
    """
    Evaluate the agent on the evaluation environments for at least n episodes.

    Args:
        n: The minimum number of episodes to evaluate.
        sample_fn: The function to call to sample actions from the agent by passing in the observations
        eval_envs: The evaluation environments.

    Returns:
        A dictionary containing the evaluation results.
    """

    with torch.no_grad():
        eval_metrics = defaultdict(list)
        obs, info = eval_envs.reset()
        eps_count = 0
        try:
            max_episode_steps = find_max_episode_steps_value(eval_envs)
        except:
            max_episode_steps = 150
        pbar = tqdm(total=n*max_episode_steps, desc="Evaluating")
        while eps_count < n:
            action = sample_fn(obs)
            obs, _, _, truncated, info = eval_envs.step(action)
            # note as there are no partial resets, truncated is True for all environments at the same time
            if truncated.any():
                if isinstance(info["final_info"], dict):
                    for k, v in info["final_info"]["episode"].items():
                        eval_metrics[k].append(v.float().cpu().numpy())
                else:
                    for final_info in info["final_info"]:
                        for k, v in final_info["episode"].items():
                            eval_metrics[k].append(v)
                eps_count += eval_envs.num_envs
            pbar.update(eval_envs.num_envs)
    for k in eval_metrics.keys():
        eval_metrics[k] = np.stack(eval_metrics[k])
        if eval_metrics[k].ndim > 1 and eval_metrics[k].shape[0] == 1:
            eval_metrics[k] = eval_metrics[k].squeeze(0)
    return eval_metrics
