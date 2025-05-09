import torch
from aligned_calibration_dataset import create_aligned_calibration_traj_dataloader
from agents import SACTransferAgent, AgentAligner
import gymnasium as gym
import tyro
from utils import Args, Logger
from mani_skill.utils.wrappers import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils.wrappers.record import RecordEpisode
import time
from mani_skill.utils import gym_utils
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from agents import ActorCriticAgent
from collections import defaultdict


def evaluate(agent: ActorCriticAgent, eval_envs: gym.Env, args: Args, logger: Logger, global_step: int):
    agent.actor.eval()
    stime = time.perf_counter()
    eval_obs, _ = eval_envs.reset()
    eval_metrics = defaultdict(list)
    num_episodes = 0
    for _ in range(args.num_eval_steps):
        with torch.no_grad():
            eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(agent.get_eval_action(eval_obs))
            if "final_info" in eval_infos:
                mask = eval_infos["_final_info"]
                num_episodes += mask.sum()
                for k, v in eval_infos["final_info"]["episode"].items():
                    eval_metrics[k].append(v)
    eval_metrics_mean = {}
    for k, v in eval_metrics.items():
        mean = torch.stack(v).float().mean()
        eval_metrics_mean[k] = mean
        if logger is not None:
            logger.add_scalar(f"eval/{k}", mean, global_step)
    pbar.set_description(
        f"success_once: {eval_metrics_mean['success_once']:.2f}, "
        f"return: {eval_metrics_mean['return']:.2f}"
    )
    if logger is not None:
        eval_time = time.perf_counter() - stime
        logger.add_scalar("time/eval_time", eval_time, global_step)
    agent.actor.train()

    if args.save_model:
        model_path = f"{args.save_model_dir}/{run_name}/ckpt_{global_step}.pt"
        agent.save_model(model_path)

    return eval_metrics_mean


def setup_envs(args: Args):
    env_kwargs = dict(obs_mode="state", render_mode="rgb_array", sim_backend="gpu")
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    source_envs = gym.make(args.env_id, robot_uids=args.source_robot, num_envs=args.num_eval_envs, reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)
    target_envs = gym.make(args.env_id, robot_uids=args.target_robot, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, human_render_camera_configs=dict(shader_pack="default"), **env_kwargs)
    if isinstance(source_envs.action_space, gym.spaces.Dict):
        source_envs = FlattenActionSpaceWrapper(source_envs)
        target_envs = FlattenActionSpaceWrapper(target_envs)
    if args.capture_video or args.save_trajectory:
        eval_output_dir = f"runs/{run_name}/videos"
        print(f"Saving eval trajectories/videos to {eval_output_dir}")
        target_envs = RecordEpisode(target_envs, output_dir=eval_output_dir, save_trajectory=args.save_trajectory, save_video=args.capture_video, trajectory_name="trajectory", max_steps_per_video=args.num_eval_steps, video_fps=30, wandb_video_freq=args.wandb_video_freq if args.track else 0)
    source_envs = ManiSkillVectorEnv(source_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=False)
    target_envs = ManiSkillVectorEnv(target_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    return source_envs, target_envs, env_kwargs


if __name__ == "__main__":
    args = tyro.cli(Args)
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
        config["source_env_cfg"] = dict(**env_kwargs, robot_uids=args.source_robot, num_envs=args.num_eval_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.eval_partial_reset)
        config["target_env_cfg"] = dict(**env_kwargs, robot_uids=args.target_robot, num_envs=args.num_eval_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.eval_partial_reset)
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
    if args.source_trained_model_path is not None:
        source_agent.load_model(args.source_trained_model_path)
    target_agent = SACTransferAgent(target_envs, device, args)
    
    agent_aligner = AgentAligner(device, args, source_agent, target_agent)

    # Create dataloader using the helper function
    dataloader = create_aligned_calibration_traj_dataloader(
        source_path=args.alignment_source_traj_path,
        target_path=args.alignment_target_traj_path,
        batch_size=args.alignment_batch_size,
        device=device,
        load_count=10,
        steps_per_epoch=args.alignment_samples//args.alignment_batch_size,
        shuffle_goals=False,
        normalize_states=False
    )
    print(f"Total batches per epoch: {len(dataloader)}")

    pbar = tqdm(range(args.alignment_samples), desc="Aligning agents")
    # Example training loop
    for batch_idx, batch in enumerate(dataloader):
        global_step = batch_idx * args.alignment_batch_size
        if args.eval_freq > 0 and (global_step - args.alignment_batch_size) // args.eval_freq < global_step // args.eval_freq:
            evaluate(target_agent, target_envs, args, logger, global_step)

        update_time = time.perf_counter()
        agent_aligner.update(batch.to_source_replay_buffer_sample(), 
                             batch.to_target_replay_buffer_sample(),
                             global_step)
        update_time = time.perf_counter() - update_time

        if (global_step - args.alignment_batch_size) // args.log_freq < global_step // args.log_freq:
            agent_aligner.log_losses(logger, global_step)
            logger.add_scalar("time/align_update_time", update_time, global_step)
        pbar.update(args.alignment_batch_size)
    
    if args.save_model:
        model_path = f"{args.save_model_dir}/{run_name}/final_ckpt.pt"
        target_agent.save_model(model_path)
    writer.close()
    source_envs.close()
    target_envs.close()
