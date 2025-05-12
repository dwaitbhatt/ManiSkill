
from collections import defaultdict
import os
import random
import time

import tqdm

from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from utils import Args, Logger
from replay_buffer import ReplayBuffer
from agents import ActorCriticAgent, TD3Agent, SACAgent, SACTransferAgent, JointTrainer

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import tyro


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
        cumulative_times["eval_time"] += eval_time
        logger.add_scalar("time/eval_time", eval_time, global_step)
    agent.actor.train()

    if args.save_model:
        model_path = f"{args.save_model_dir}/{run_name}/ckpt_{global_step}.pt"
        agent.save_model(model_path)

    return eval_metrics_mean


def setup_envs(args: Args, run_name: str, robot_uids: str):
    ####### Environment setup #######
    env_kwargs = dict(obs_mode="state", render_mode="rgb_array", sim_backend="gpu", robot_uids=robot_uids)
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1, reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, human_render_camera_configs=dict(shader_pack="default"), **env_kwargs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    if args.capture_video or args.save_trajectory:
        eval_output_dir = f"runs/{run_name}/videos_{robot_uids}"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos_{robot_uids}"
        print(f"Saving eval trajectories/videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x : (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(envs, output_dir=f"runs/{run_name}/train_videos_{robot_uids}", save_trajectory=False, save_video_trigger=save_video_trigger, max_steps_per_video=args.num_steps, video_fps=30)
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=args.save_trajectory, save_video=args.capture_video, trajectory_name="trajectory", max_steps_per_video=args.num_eval_steps, video_fps=30, wandb_video_freq=args.wandb_video_freq)
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    return envs, eval_envs, env_kwargs


def rollout_and_fill_buffer(agent: ActorCriticAgent, envs: gym.Env, rb: ReplayBuffer, global_step: int, starting_obs: torch.Tensor):
    obs = starting_obs
    for local_step in range(args.steps_per_env):
        global_step += 1 * args.num_envs

        # ALGO LOGIC: put action logic here
        if not learning_has_started:
            actions = torch.tensor(envs.action_space.sample(), dtype=torch.float32, device=device)
        else:
            actions = agent.sample_action(obs)
            actions = actions.detach()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        real_next_obs = next_obs.clone()
        if args.bootstrap_at_done == 'never':
            need_final_obs = torch.ones_like(terminations, dtype=torch.bool)
            stop_bootstrap = truncations | terminations # always stop bootstrap when episode ends
        else:
            if args.bootstrap_at_done == 'always':
                need_final_obs = truncations | terminations # always need final obs when episode ends
                stop_bootstrap = torch.zeros_like(terminations, dtype=torch.bool) # never stop bootstrap
            else: # bootstrap at truncated
                need_final_obs = truncations & (~terminations) # only need final obs when truncated and not terminated
                stop_bootstrap = terminations # only stop bootstrap when terminated, don't stop when truncated
        if "final_info" in infos:
            final_info = infos["final_info"]
            done_mask = infos["_final_info"]
            real_next_obs[need_final_obs] = infos["final_observation"][need_final_obs]
            for k, v in final_info["episode"].items():
                logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)

        rb.add(obs, real_next_obs, actions, rewards, stop_bootstrap)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
    return global_step, obs


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.grad_steps_per_iteration = int(args.training_freq * args.utd)
    args.steps_per_env = args.training_freq // args.num_envs
    if args.exp_name is None:
        args.exp_name = f"{args.algorithm.lower()}"
        run_name = f"{args.env_id}__{args.exp_name}__{args.source_robot}__{args.target_robot}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

    source_envs, source_eval_envs, source_env_kwargs = setup_envs(args, run_name, args.source_robot)
    target_envs, target_eval_envs, target_env_kwargs = setup_envs(args, run_name, args.target_robot)
    max_episode_steps = gym_utils.find_max_episode_steps_value(source_envs._env)
    logger = None
    if not args.evaluate:
        print("Running training")
        if args.track:
            import wandb
            config = vars(args)
            config["source_env_cfg"] = dict(**source_env_kwargs, num_envs=args.num_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            config["source_eval_env_cfg"] = dict(**source_env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=False)
            config["target_env_cfg"] = dict(**source_env_kwargs, num_envs=args.num_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            config["target_eval_env_cfg"] = dict(**source_env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=False)
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
    else:
        print("Running evaluation")

    source_envs.single_observation_space.dtype = np.float32
    rb_source = ReplayBuffer(
        env=source_envs,
        num_envs=args.num_envs,
        buffer_size=args.buffer_size,
        storage_device=torch.device(args.buffer_device),
        sample_device=device
    )
    target_envs.single_observation_space.dtype = np.float32
    rb_target = ReplayBuffer(
        env=target_envs,
        num_envs=args.num_envs,
        buffer_size=args.buffer_size,
        storage_device=torch.device(args.buffer_device),
        sample_device=device
    )

    # TRY NOT TO MODIFY: start the game
    source_obs, info = source_envs.reset(seed=args.seed) # in Gymnasium, seed is given to reset() instead of seed()
    source_eval_obs, _ = source_eval_envs.reset(seed=args.seed)
    target_obs, _ = target_envs.reset(seed=args.seed)
    target_eval_obs, _ = target_eval_envs.reset(seed=args.seed)
    
    global_step = 0
    global_update = 0
    learning_has_started = False

    source_agent = SACTransferAgent(source_envs, device, args, name="source")
    target_agent = SACTransferAgent(target_envs, device, args, name="target")
    joint_trainer = JointTrainer(device, args, source_agent, target_agent)

    print(f"\n\n\n############# Network architecture: ##############\n{source_agent}\n\n\n")

    global_steps_per_iteration = args.num_envs * (args.steps_per_env)
    pbar = tqdm.tqdm(range(args.total_timesteps))
    cumulative_times = defaultdict(float)

    while global_step < args.total_timesteps:
        if args.eval_freq > 0 and (global_step - args.training_freq) // args.eval_freq < global_step // args.eval_freq:
            # evaluate
            evaluate(source_agent, source_eval_envs, args, logger, global_step)
            evaluate(target_agent, target_eval_envs, args, logger, global_step)
            if args.evaluate:
                break

        # Collect samples from environments
        rollout_time = time.perf_counter()
        global_step, source_obs = rollout_and_fill_buffer(source_agent, source_envs, rb_source, global_step, source_obs)
        global_step, target_obs = rollout_and_fill_buffer(target_agent, target_envs, rb_target, global_step, target_obs)
        rollout_time = time.perf_counter() - rollout_time

        cumulative_times["rollout_time"] += rollout_time
        pbar.update(args.num_envs * args.steps_per_env)

        # ALGO LOGIC: training.
        if global_step < args.learning_starts:
            continue

        update_time = time.perf_counter()
        learning_has_started = True
        
        global_update = joint_trainer.update(rb_source, rb_target, global_update, global_step)
        
        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time

        # Log training-related data
        if (global_step - args.training_freq) // args.log_freq < global_step // args.log_freq:
            source_agent.log_losses(logger, global_step)
            target_agent.log_losses(logger, global_step)
            logger.add_scalar("time/update_time", update_time, global_step)
            logger.add_scalar("time/rollout_time", rollout_time, global_step)
            logger.add_scalar("time/rollout_fps", global_steps_per_iteration / rollout_time, global_step)
            for k, v in cumulative_times.items():
                logger.add_scalar(f"time/total_{k}", v, global_step)
            logger.add_scalar("time/total_rollout+update_time", cumulative_times["rollout_time"] + cumulative_times["update_time"], global_step)

    if not args.evaluate and args.save_model:
        model_path = f"{args.save_model_dir}/{run_name}/source_final_ckpt.pt"
        source_agent.save_model(model_path)
        model_path = f"{args.save_model_dir}/{run_name}/target_final_ckpt.pt"
        target_agent.save_model(model_path)
    writer.close()
    source_envs.close()
    target_envs.close()
