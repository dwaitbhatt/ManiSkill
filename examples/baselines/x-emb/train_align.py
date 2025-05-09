import torch
from aligned_calibration_dataset import create_aligned_calibration_traj_dataloader
from agents import SACTransferAgent, AgentAligner
import gymnasium as gym
import tyro
from utils import Args, Logger
from mani_skill.utils.wrappers import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
import time
from mani_skill.utils import gym_utils
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter


def setup_envs(args: Args):
    env_kwargs = dict(obs_mode="state", render_mode="rgb_array", sim_backend="gpu", robot_uids=args.robot)
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    envs = gym.make("XembCalibration-v1", num_envs=args.num_envs if not args.evaluate else 1, reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=False)
    return envs, env_kwargs


def main():
    args = tyro.cli(Args)
    args.grad_steps_per_iteration = int(args.training_freq * args.utd)
    args.steps_per_env = args.training_freq // args.num_envs
    if args.exp_name is None:
        args.exp_name = f"{args.algorithm.lower()}"
        run_name = f"{args.env_id}__{args.exp_name}__{args.robot}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name
    run_name += f"__align"

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

    envs, env_kwargs = setup_envs(args)
    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)

    print("\nRunning agent alignment")
    if args.track:
        import wandb
        config = vars(args)
        config["env_cfg"] = dict(**env_kwargs, num_envs=args.num_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.partial_reset)
        config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=False)
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=False,
            config=config,
            name=run_name,
            save_code=True,
            group=args.wandb_group,
            tags=[args.algorithm.lower(), "walltime_efficient"]
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    logger = Logger(log_wandb=args.track, tensorboard=writer)


    # Initialize agents
    source_agent = SACTransferAgent(envs, device, args)
    target_agent = SACTransferAgent(envs, device, args)
    print(f"Total batches per epoch: {len(dataloader)}")
    
    agent_aligner = AgentAligner(envs, device, args, source_agent, target_agent)

    # Create dataloader using the helper function
    dataloader = create_aligned_calibration_traj_dataloader(
        source_path=args.alignment_source_traj_path,
        target_path=args.alignment_target_traj_path,
        batch_size=128,
        device=device,
        load_count=10,
        steps_per_epoch=args.alignment_steps,
        shuffle_goals=False,
        normalize_states=False
    )

    # Example training loop
    for batch_idx, batch in enumerate(dataloader):
        agent_aligner.update(batch.to_source_replay_buffer_sample(), 
                             batch.to_target_replay_buffer_sample(),
                             batch_idx)

if __name__ == "__main__":
    main()