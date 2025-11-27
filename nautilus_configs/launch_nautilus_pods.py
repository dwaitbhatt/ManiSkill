#!/usr/bin/env python3

"""
Script to launch multiple Nautilus pods with different random seeds.

This script can be run with command-line arguments:
./launch_nautilus_pods.py --algo td3 --robot xarm6_robotiq --env-id PickCube-v1 --exp-suffix test --total-timesteps 1_000_000
"""

import os
import yaml
import random
import subprocess
import tempfile
import tyro
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Annotated, Literal

class Algorithm(str, Enum):
    SAC = "SAC"
    PPO = "PPO"
    DUMMY = "DUMMY"
    ACT = "ACT"

@dataclass
class NautilusPodConfig:
    """Configuration for launching Nautilus pods with different random seeds."""
    
    algo: Annotated[Literal["SAC", "PPO", "DUMMY", "ACT"], 
                    tyro.conf.arg(
                        name="algo", 
                        help="Algorithm to use (SAC, PPO, DUMMY, case-sensitive)",
                        aliases=["-a"]
                    )] = "ACT"
    
    robot: Annotated[str, 
                    tyro.conf.arg(
                        name="robot", 
                        help="Robot to use (e.g., xarm6_robotiq). For alignment, use robot1_to_robot2 (e.g., panda_to_xarm6_robotiq)",
                        aliases=["-r"]
                    )] = "xarm6_robotiq"
    
    env_id: Annotated[str, 
                    tyro.conf.arg(
                        name="env_id", 
                        help="Environment ID (e.g., PickCube-v1)",
                        aliases=["-e"]
                    )] = "PickCube-v1"

    control_mode: Annotated[str, 
                            tyro.conf.arg(
                                name="control_mode", 
                                help="Control mode to use (e.g., pd_joint_vel)",
                                aliases=["-c"]
                            )] = "pd_joint_vel"
    
    exp_suffix: Annotated[str, 
                        tyro.conf.arg(
                            name="exp_suffix", 
                            help="Suffix for the experiment name (e.g., 'newnets_fixed_long')",
                            aliases=["-x"]
                        )] = ""
    
    total_timesteps: Annotated[str, 
                              tyro.conf.arg(
                                  name="total_timesteps", 
                                  help="Total timesteps (e.g., 1_000_000)",
                                  aliases=["-t"]
                              )] = "200_000"
    
    branch: Annotated[str, 
                    tyro.conf.arg(
                        name="branch", 
                        help="Git branch to use for the experiment",
                        aliases=["-b"]
                    )] = "act"
    
    num_pods: Annotated[int, 
                        tyro.conf.arg(
                            name="num_pods", 
                            help="Number of pods to launch",
                            aliases=["-n"]
                        )] = 1

    jobs: Annotated[bool, 
                    tyro.conf.arg(
                        name="jobs", 
                        help="Whether to launch jobs instead of pods",
                        aliases=["-j"]
                    )] = False
    
    save_models: Annotated[bool, 
                          tyro.conf.arg(
                              name="save_models", 
                              help="Whether to save models to persistent volume",
                              aliases=["-s"]
                          )] = False

    yaml_file: Optional[str] = None
    """YAML kubernetes config file to use as template"""
    
    wandb_entity: str = "ucsd_erl"
    """Wandb entity (team) name"""
    
    wandb_project: str = "maniskill_experiments"
    """Wandb project name"""
    
    extra: str = ""
    """Extra command line arguments to pass to the training script, which will override default values.
    Format: '--arg1 value1 --arg2 value2' or '--arg1=value1 --arg2=value2'."""


def generate_experiment_name(env_id: str, robot: str, control_mode: str, algo: Literal["SAC", "PPO", "DUMMY", "ACT"], suffix: str = "") -> str:
    """Generate a standard experiment name from environment, robot, algorithm, and optional suffix."""
    env_name = env_id.split('-')[0]
    exp_name = f"{env_name}-{robot}-{control_mode}-{algo}"
    if suffix:
        exp_name = f"{exp_name}-{suffix}"
    
    return exp_name


def generate_command(algo: Literal["SAC", "PPO", "DUMMY", "ACT"], robot: str, env_id: str, control_mode: str, exp_name: str, 
                    total_timesteps: str, seed: int, branch: str,
                    wandb_entity: str, wandb_project: str, run_name: str,
                    extra_cmd_args: str = "",
                    save_models: bool = False) -> str:
    """Generate the command based on the algorithm and parameters."""
    timestamp_log = "$(date +%Y-%m-%d_%H-%M-%S)"
    
    # Git commands to reset to the specified branch
    git_commands = f'''git fetch origin {branch} && git reset --hard origin/{branch}'''

    # Command to copy saved models to persistent volume
    copy_saved_models_command = ":"
    if save_models:
        copy_saved_models_command = f'''printf \"\nCopying saved models to persistent volume...\" && \\
                                    mkdir -p /pers_vol/dwait/saved_models/ && \\
                                    cp -r runs/ /pers_vol/dwait/saved_models/'''
    
    # Parse extra command arguments into a dictionary
    extra_args_dict = {}
    if extra_cmd_args:
        for arg in extra_cmd_args.split():
            if arg.startswith('--'):
                key = arg[2:]  # Remove leading --
                if '=' in key:
                    key, value = key.split('=', 1)
                    extra_args_dict[key] = value
                else:
                    # Next argument might be a value or another flag
                    extra_args_dict[key] = True
            elif extra_args_dict:
                # If the previous item was a key, and this isn't a flag, it's a value
                last_key = list(extra_args_dict.keys())[-1]
                if extra_args_dict[last_key] is True:  # Only update if it was a placeholder
                    extra_args_dict[last_key] = arg
    
    # Common wandb arguments for all algorithms
    wandb_args = {
        'wandb-entity': wandb_entity,
        'wandb-project-name': wandb_project,
        'exp_name': run_name,
        'wandb_group': exp_name,
        'track': True,
    }
    
    if algo == Algorithm.SAC:
        cmd_args = {
            **wandb_args,
            'env_id': env_id,
            'robot': robot,
            'control_mode': control_mode,
            'gamma': 0.95,
            'num_envs': '128',
            'training_freq': '128',
            'num_eval_steps': '100',
            'eval_freq': '100_000',
            'total_timesteps': total_timesteps,
            'wandb_video_freq': '2',
        }
        cmd_args.update(extra_args_dict)
        args_str = ' '.join([f'--{k}={v}' if v is not True else f'--{k}' for k, v in cmd_args.items()])
        main_cmd = f'''echo y | python examples/baselines/sac/sac.py {args_str} \\
                    > /pers_vol/dwait/logs/{timestamp_log}-{algo}.log'''

    elif algo == Algorithm.PPO:
        # Base arguments for PPO
        cmd_args = {
            **wandb_args,
            'env_id': env_id,
            'robot_uids': robot,
            'control_mode': control_mode,
            'seed': str(seed),
            'num_envs': '512',
            'num_eval_envs': '8',
            'eval_freq': '10',
            'total_timesteps': total_timesteps,
            'num_steps': '50',
            'num_minibatches': '32',
            'num_eval_steps': '50',
            'gamma': '0.8',
            'update_epochs': '4',
            'wandb_video_freq': '2',
        }
        
        # Override with any extra arguments
        cmd_args.update(extra_args_dict)
        
        # Build the command string
        args_str = ' '.join([f'--{k}={v}' if v is not True else f'--{k}' for k, v in cmd_args.items()])
        main_cmd = f'''pip install tensordict && echo y | python examples/baselines/ppo/ppo_fast.py {args_str} \\
                    > /pers_vol/dwait/logs/{timestamp_log}-{algo}.log''' 
    
    elif algo == Algorithm.ACT:
        cmd_args = {
            **wandb_args,
            'env_id': env_id,
            'robot_uid': robot,
            'demo_path': f'/pers_vol/dwait/saved_demos/demos/{env_id}/motionplanning/{robot}/trajectory.state.pd_joint_vel.physx_cpu.h5',
            'control_mode': control_mode,
            'sim_backend': 'gpu',
            'num_demos': '1000',
            'max_episode_steps': '100',
            'num_eval_envs': '30',
            'total_iters': total_timesteps,
            'log_freq': '100',
            'eval_freq': '5000',
            'wandb_video_freq': '1',
            'demo_type': 'motionplanning'
        }
        cmd_args.update(extra_args_dict)

        args_str = ' '.join([f'--{k}={v}' if v is not True else f'--{k}' for k, v in cmd_args.items()])
        main_cmd = f'''pip install torchvision diffusers && \\
                    echo y | python examples/baselines/act/train.py {args_str} \\
                    > /pers_vol/dwait/logs/{timestamp_log}-{algo}.log'''

    elif algo == Algorithm.DUMMY:
        main_cmd = "sleep infinity"

    return f'''{git_commands} && \\
            {main_cmd} && \\
            {copy_saved_models_command}'''


def launch_pod(yaml_file: str, pod_name: str, command: str, jobs: bool = False) -> None:
    """Modify the YAML and launch the pod with kubectl."""
    with open(yaml_file, 'r') as f:
        pod_config = yaml.safe_load(f)
    
    if jobs:
        pod_config['metadata']['name'] = pod_name
        pod_config['spec']['template']['spec']['containers'][0]['args'] = [command]
    else:
        pod_config['metadata']['name'] = pod_name    
        pod_config['spec']['containers'][0]['args'] = [command]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
        yaml.dump(pod_config, temp_file)
        temp_file_path = temp_file.name
    
    try:
        subprocess.run(['kubectl', 'apply', '-f', temp_file_path], check=True)
        print(f"Successfully launched: {pod_name}\n")
    except subprocess.CalledProcessError as e:
        print(f"Failed to launch {pod_name}: {e}\n")
    finally:
        os.unlink(temp_file_path)


def main() -> None:
    config = tyro.cli(NautilusPodConfig)

    exp_name = generate_experiment_name(config.env_id, config.robot, config.control_mode, config.algo, config.exp_suffix)
    
    timestamp = datetime.now().strftime("%m%d-%H%M")

    nautilus_type = "job" if config.jobs and config.algo != Algorithm.DUMMY else "pod"
    if config.yaml_file is None:
        yaml_file = f"nautilus_configs/db-maniskill-{nautilus_type}.yaml"
    else:
        yaml_file = config.yaml_file
    
    print(f"Launching {nautilus_type}s with the following configuration:")
    print(f"  Algorithm: {config.algo}")
    print(f"  Robot: {config.robot}")
    print(f"  Environment: {config.env_id}")
    print(f"  Experiment name: {exp_name}")
    print(f"  Total timesteps: {config.total_timesteps}")
    print(f"  Git branch: {config.branch}")
    print(f"  YAML template: {yaml_file}")
    print(f"  Extra command line arguments: {config.extra}" if config.extra else "")

    if config.algo == Algorithm.DUMMY:
        pod_name = f"maniskill-{config.branch}-dummy"
        command = "sleep infinity"
        launch_pod(yaml_file, pod_name, command)
        print(f"Launched dummy pod for {config.branch} branch. Remember to delete it once done.")
        return
    else:
        print(f"  Wandb entity: {config.wandb_entity}")
        print(f"  Wandb project: {config.wandb_project}")
        print(f"  Wandb group: {exp_name}")
        print()

    for i in range(config.num_pods):
        seed = random.randint(1, 10000)
        
        pod_name = f"maniskill-{timestamp}-{config.branch}-{config.exp_suffix}-seed{seed}"        
        pod_name = pod_name.replace('_', '-').lower()
        
        # Run name format: [env]_[robot]_[algo]_[suffix]_seed[seed]
        env_short = config.env_id.split('-')[0].lower()  # Use just the env name without version
        run_name = f"{env_short}_{config.robot}_{config.control_mode}_{config.algo}_{config.exp_suffix}_seed{seed}"
        
        command = generate_command(
            config.algo, config.robot, config.env_id, config.control_mode, exp_name, 
            config.total_timesteps, seed, config.branch,
            config.wandb_entity, config.wandb_project, run_name,
            extra_cmd_args=config.extra,
            save_models=config.save_models
        )
        
        print(f"Launching {nautilus_type} {i+1}/{config.num_pods} with seed {seed}...")
        launch_pod(yaml_file, pod_name, command, config.jobs)

if __name__ == "__main__":
    main()