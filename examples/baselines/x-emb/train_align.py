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
from tqdm.auto import tqdm


def setup_envs(args: Args):
    env_kwargs = dict(obs_mode="state", render_mode="rgb_array", sim_backend="gpu")
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    source_envs = gym.make(args.env_id, robot_uids=args.source_robot, num_envs=args.num_envs if not args.evaluate else 1, reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)
    target_envs = gym.make(args.env_id, robot_uids=args.target_robot, num_envs=args.num_envs if not args.evaluate else 1, reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)
    if isinstance(source_envs.action_space, gym.spaces.Dict):
        source_envs = FlattenActionSpaceWrapper(source_envs)
        target_envs = FlattenActionSpaceWrapper(target_envs)
    source_envs = ManiSkillVectorEnv(source_envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=False)
    target_envs = ManiSkillVectorEnv(target_envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=False)
    return source_envs, target_envs, env_kwargs


def main():
    args = tyro.cli(Args)
    args.grad_steps_per_iteration = int(args.training_freq * args.utd)
    args.steps_per_env = args.training_freq // args.num_envs
    if args.exp_name is None:
        args.exp_name = f"{args.algorithm.lower()}"
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name
    run_name = f"align_{args.source_robot}_{args.target_robot}__{run_name}"

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

    source_envs, target_envs, env_kwargs = setup_envs(args)
    max_episode_steps = gym_utils.find_max_episode_steps_value(source_envs._env)

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
            tags=[args.algorithm.lower(), "transfer_learning"]
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    logger = Logger(log_wandb=args.track, tensorboard=writer)


    # Initialize agents
    source_agent = SACTransferAgent(source_envs, device, args)
    target_agent = SACTransferAgent(target_envs, device, args)
    
    agent_aligner = AgentAligner(device, args, source_agent, target_agent)

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
    print(f"Total batches per epoch: {len(dataloader)}")

    # Example training loop
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Aligning agents")):
        update_time = time.perf_counter()
        agent_aligner.update(batch.to_source_replay_buffer_sample(), 
                             batch.to_target_replay_buffer_sample(),
                             batch_idx)
        update_time = time.perf_counter() - update_time

        if (batch_idx - 1) // args.log_freq < batch_idx // args.log_freq:
            agent_aligner.log_losses(logger, batch_idx)
            logger.add_scalar("time/align_update_time", update_time, batch_idx)

if __name__ == "__main__":
    main()