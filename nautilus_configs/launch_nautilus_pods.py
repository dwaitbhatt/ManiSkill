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
from dataclasses import dataclass, field
from enum import Enum

# Define a case-insensitive algorithm enum
class Algorithm(str, Enum):
    SAC = "sac"
    TD3 = "td3"
    PPO = "ppo"
    SAC_LATENT = "sac_latent"

    @classmethod
    def _missing_(cls, value):
        # Handle case-insensitive matching
        if not isinstance(value, str):
            return None
        
        # Convert the input value to lowercase for comparison
        value_lower = value.lower()
        for member in cls:
            if member.value.lower() == value_lower:
                return member
        return None


@dataclass
class NautilusPodConfig:
    """Configuration for launching Nautilus pods with different random seeds."""
    
    # Required parameters
    algo: Algorithm = "td3"
    """Algorithm to use (sac, td3, ppo, case-insensitive)"""
    
    robot: str = "xarm6_robotiq"
    """Robot to use (e.g., xarm6_robotiq)"""
    
    env_id: str = "PickCube-v1"
    """Environment ID (e.g., PickCube-v1)"""
    
    exp_suffix: str = ""
    """Suffix for the experiment name (e.g., 'newnets_fixed_long')"""
    
    total_timesteps: str = "1_000_000"
    """Total timesteps (e.g., 1_000_000)"""
    
    # Optional parameters
    branch: str = "xemb-transfer"
    """Git branch to use for the experiment"""
    
    num_pods: int = 5
    """Number of pods to launch"""
    
    yaml_file: str = "nautilus_configs/db-maniskill-pod.yaml"
    """YAML kubernetes config file to use as template"""
    
    # Wandb parameters
    wandb_entity: str = "ucsd_erl"
    """Wandb entity (team) name"""
    
    wandb_project: str = "maniskill_experiments"
    """Wandb project name"""
    
    wandb_prefix: str = "batch1"
    """Prefix for individual run names in wandb (optional)"""


def generate_experiment_name(env_id: str, robot: str, algo: Algorithm, suffix: str = "") -> str:
    """Generate a standard experiment name from environment, robot, algorithm, and optional suffix."""
    # Extract the environment name without version
    env_name = env_id.split('-')[0]
    
    # Construct the base experiment name
    exp_name = f"{env_name}_{robot}_{algo.value}"
    
    # Add suffix if provided
    if suffix:
        exp_name = f"{exp_name}_{suffix}"
    
    return exp_name


def generate_command(algo: Algorithm, robot: str, env_id: str, exp_name: str, 
                    total_timesteps: str, seed: int, branch: str,
                    wandb_entity: str, wandb_project: str, run_name: str) -> str:
    """Generate the command based on the algorithm and parameters."""
    timestamp_log = "$(date +%Y-%m-%d_%H-%M-%S)"
    
    # Start with git commands to reset to the specified branch
    git_commands = f'''git fetch origin {branch} && git reset --hard origin/{branch} && '''
    
    # Common wandb arguments for all algorithms
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


def launch_pod(yaml_file: str, pod_name: str, command: str) -> None:
    """Modify the YAML and launch the pod with kubectl."""
    with open(yaml_file, 'r') as f:
        pod_config = yaml.safe_load(f)
    
    # Update the pod name
    pod_config['metadata']['name'] = pod_name
    
    # Update the command
    pod_config['spec']['containers'][0]['args'] = [command]
    
    # Write modified YAML to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
        yaml.dump(pod_config, temp_file)
        temp_file_path = temp_file.name
    
    # Launch pod with kubectl
    try:
        subprocess.run(['kubectl', 'apply', '-f', temp_file_path], check=True)
        print(f"Successfully launched pod: {pod_name}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to launch pod {pod_name}: {e}")
    finally:
        # Clean up temporary file
        os.unlink(temp_file_path)


def main() -> None:
    # Parse arguments with tyro, directly supporting config file loading
    config = tyro.cli(NautilusPodConfig)
    
    # Generate the experiment name from the provided parameters
    exp_name = generate_experiment_name(config.env_id, config.robot, config.algo, config.exp_suffix)
    
    # Generate a timestamp for pod naming
    timestamp = datetime.now().strftime("%m%d-%H%M")
    
    # Display configuration information
    print(f"Launching {config.num_pods} pods with the following configuration:")
    print(f"  Algorithm: {config.algo.value}")
    print(f"  Robot: {config.robot}")
    print(f"  Environment: {config.env_id}")
    print(f"  Experiment name: {exp_name}")
    print(f"  Total timesteps: {config.total_timesteps}")
    print(f"  Git branch: {config.branch}")
    print(f"  YAML template: {config.yaml_file}")
    print(f"  Wandb entity: {config.wandb_entity}")
    print(f"  Wandb project: {config.wandb_project}")
    print(f"  Wandb group: {exp_name}")
    print()
    
    # Launch multiple pods with different seeds
    for i in range(config.num_pods):
        seed = random.randint(1, 10000)
        
        # Create a meaningful pod name for Kubernetes
        # Include timestamp to ensure uniqueness and branch for tracking
        pod_name = f"maniskill-{timestamp}-{config.branch}-{config.algo.value}-seed{seed}"
        
        # Make sure the pod name is compliant with Kubernetes naming rules
        # Only lowercase alphanumeric characters, '-', and '.'
        pod_name = pod_name.replace('_', '-').lower()
        
        # Create a meaningful run name for W&B
        # Format: [prefix]_[env]_[robot]_[algo]_seed[seed]
        prefix = f"{config.wandb_prefix}_" if config.wandb_prefix else ""
        env_short = config.env_id.split('-')[0].lower()  # Use just the env name without version
        run_name = f"{prefix}{env_short}_{config.robot}_{config.algo.value}_seed{seed}"
        
        # Generate the command based on algorithm
        command = generate_command(
            config.algo, config.robot, config.env_id, exp_name, 
            config.total_timesteps, seed, config.branch,
            config.wandb_entity, config.wandb_project, run_name
        )
        
        # Launch the pod
        launch_pod(config.yaml_file, pod_name, command)
        print(f"Launched pod {i+1}/{config.num_pods} with seed {seed}")
        print(f"  W&B run name: {run_name}")
        print(f"  W&B group: {exp_name}")
        print()


if __name__ == "__main__":
    main()
