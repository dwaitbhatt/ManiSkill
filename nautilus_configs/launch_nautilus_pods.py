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
from typing import Optional

# Define a case-insensitive algorithm enum
class Algorithm(str, Enum):
    SAC = "SAC"
    TD3 = "TD3"
    PPO = "PPO"
    SAC_LATENT = "SAC_LATENT"
    DUMMY = "DUMMY"

@dataclass
class NautilusPodConfig:
    """Configuration for launching Nautilus pods with different random seeds."""
    
    algo: Algorithm = "td3"
    """Algorithm to use (sac, td3, ppo, sac_latent, dummy, case-insensitive)"""
    
    robot: str = "xarm6_robotiq"
    """Robot to use (e.g., xarm6_robotiq)"""
    
    env_id: str = "PickCube-v1"
    """Environment ID (e.g., PickCube-v1)"""
    
    exp_suffix: str = ""
    """Suffix for the experiment name (e.g., 'newnets_fixed_long')"""
    
    total_timesteps: str = "1_000_000"
    """Total timesteps (e.g., 1_000_000)"""
    
    branch: str = "xemb-transfer"
    """Git branch to use for the experiment"""
    
    num_pods: int = 4
    """Number of pods to launch"""

    jobs: bool = False
    """Whether to launch jobs instead of pods"""
    
    yaml_file: Optional[str] = None
    """YAML kubernetes config file to use as template"""
    
    wandb_entity: str = "ucsd_erl"
    """Wandb entity (team) name"""
    
    wandb_project: str = "maniskill_experiments"
    """Wandb project name"""
    
    extra: str = ""
    """Extra command line arguments to pass to the experiment"""


def generate_experiment_name(env_id: str, robot: str, algo: Algorithm, suffix: str = "") -> str:
    """Generate a standard experiment name from environment, robot, algorithm, and optional suffix."""
    env_name = env_id.split('-')[0]
    exp_name = f"{env_name}_{robot}_{algo.value}"
    if suffix:
        exp_name = f"{exp_name}_{suffix}"
    
    return exp_name


def generate_command(algo: Algorithm, robot: str, env_id: str, exp_name: str, 
                    total_timesteps: str, seed: int, branch: str,
                    wandb_entity: str, wandb_project: str, run_name: str,
                    extra_cmd_args: str = "") -> str:
    """Generate the command based on the algorithm and parameters."""
    timestamp_log = "$(date +%Y-%m-%d_%H-%M-%S)"
    
    git_commands = f'''git fetch origin {branch} && git reset --hard origin/{branch} && '''    
    wandb_args = f'''--wandb-entity {wandb_entity} --wandb-project-name {wandb_project} --exp_name {run_name} --wandb_group {exp_name}'''
    
    if algo == Algorithm.SAC or algo == Algorithm.TD3:
        return f'''{git_commands}
echo y | python examples/baselines/x-emb/train_source.py \\
       {wandb_args} \\
       --algorithm {algo.value.upper()} \\
       --env-id {env_id} \\
       --robot {robot} \\
       --control_mode pd_joint_vel \\
       --seed {seed} \\
       --num-envs 128 \\
       --training-freq 128 \\
       --num-eval-steps 100 \\
       --eval-freq 50_000 \\
       --total-timesteps {total_timesteps} \\
       --wandb-video-freq 2 \\
       --track \\
       {extra_cmd_args} \\
  > /pers_vol/dwait/logs/{timestamp_log}-{algo.value}.log'''
    
    elif algo == Algorithm.PPO:
        return f'''{git_commands}
echo y | python examples/baselines/ppo/ppo.py \\
       {wandb_args} \\
       --env_id={env_id} \\
       --robot_uids={robot} \\
       --control_mode=pd_joint_vel \\
       --seed={seed} \\
       --num_envs=512 \\
       --num_eval_envs=8 \\
       --eval_freq=10 \\
       --total_timesteps={total_timesteps} \\
       --num_steps=50 \\
       --num_minibatches=32 \\
       --num_eval_steps=50 \\
       --gamma=0.8 \\
       --update_epochs=4 \\
       --track \\
       --wandb_video_freq=2 \\
       {extra_cmd_args} \\
  > /pers_vol/dwait/logs/{timestamp_log}-{algo.value}.log'''

    elif algo == Algorithm.SAC_LATENT:
        return f'''{git_commands}
echo y | python examples/baselines/x-emb/train_source.py \\
       {wandb_args} \\
       --algorithm {algo.value.upper()} \\
       --env-id {env_id} \\
       --robot {robot} \\
       --control_mode pd_joint_vel \\
       --seed {seed} \\
       --num-envs 128 \\
       --training-freq 128 \\
       --num-eval-steps 100 \\
       --eval-freq 50_000 \\
       --total-timesteps {total_timesteps} \\
       --wandb-video-freq 2 \\
       --track \\
  > /pers_vol/dwait/logs/{timestamp_log}-{algo.value}.log'''
    
    elif algo == Algorithm.DUMMY:
        return f"{git_commands} sleep infinity"


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
        print(f"Successfully launched pod: {pod_name}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to launch pod {pod_name}: {e}")
    finally:
        os.unlink(temp_file_path)

def main() -> None:
    config = tyro.cli(NautilusPodConfig)

    exp_name = generate_experiment_name(config.env_id, config.robot, config.algo, config.exp_suffix)
    
    timestamp = datetime.now().strftime("%m%d-%H%M")

    nautilus_type = "job" if config.jobs and config.algo != Algorithm.DUMMY else "pod"
    if config.yaml_file is None:
        yaml_file = f"nautilus_configs/db-maniskill-{nautilus_type}.yaml"
    else:
        yaml_file = config.yaml_file
    
    print(f"Launching {nautilus_type}s with the following configuration:")
    print(f"  Algorithm: {config.algo.value}")
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
        
        # Run name format: [prefix]_[env]_[robot]_[algo]_seed[seed]
        env_short = config.env_id.split('-')[0].lower()  # Use just the env name without version
        run_name = f"{env_short}_{config.robot}_{config.algo.value}_{config.exp_suffix}_seed{seed}"
        
        command = generate_command(
            config.algo, config.robot, config.env_id, exp_name, 
            config.total_timesteps, seed, config.branch,
            config.wandb_entity, config.wandb_project, run_name,
            extra_cmd_args=config.extra
        )
        
        launch_pod(yaml_file, pod_name, command, config.jobs)
        print(f"Launched {nautilus_type} {i+1}/{config.num_pods} with seed {seed}\n")

if __name__ == "__main__":
    main()