import torch
import numpy as np
import h5py
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from mani_skill.utils.io_utils import load_json
from tqdm import tqdm
from replay_buffer import ReplayBufferSample


@dataclass
class AlignedTrajDataSample:
    """A class similar to ReplayBufferSample but for aligned calibration trajectory data"""
    source_obs: torch.Tensor
    source_next_obs: torch.Tensor
    source_actions: torch.Tensor
    source_dones: torch.Tensor
    source_rewards: torch.Tensor
    target_obs: torch.Tensor
    target_next_obs: torch.Tensor
    target_actions: torch.Tensor
    target_dones: torch.Tensor
    target_rewards: torch.Tensor
    current_goal_idx: torch.Tensor
    
    def to_source_replay_buffer_sample(self):
        """Convert to a standard ReplayBufferSample for source data"""
        return ReplayBufferSample(
            obs=self.source_obs,
            next_obs=self.source_next_obs,
            actions=self.source_actions,
            rewards=self.source_rewards,
            dones=self.source_dones
        )
    
    def to_target_replay_buffer_sample(self):
        """Convert to a standard ReplayBufferSample for target data"""
        return ReplayBufferSample(
            obs=self.target_obs,
            next_obs=self.target_next_obs,
            actions=self.target_actions,
            rewards=self.target_rewards,
            dones=self.target_dones
        )


def load_h5_data(data):
    """Helper function to load h5 data"""
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out


class AlignedCalibrationTrajDataset:
    """
    Dataset that loads trajectories from source and target datasets
    and organizes them by goal index for efficient sampling.
    """
    def __init__(self, source_path: str, target_path: str, device: torch.device, 
                 load_count: int = -1, normalize_states: bool = False):
        """
        Args:
            source_path: Path to source dataset h5 file
            target_path: Path to target dataset h5 file
            device: Device to load tensors on
            load_count: Number of episodes to load (-1 for all)
            normalize_states: Whether to normalize observation states
        """
        self.device = device
        
        # Load source and target datasets
        self.source_episodes = self._load_dataset(source_path, load_count, normalize_states)
        self.target_episodes = self._load_dataset(target_path, load_count, normalize_states)
        
        # Group timesteps by goal index for efficient sampling
        self.source_by_goal = self._group_by_goal_idx(self.source_episodes)
        self.target_by_goal = self._group_by_goal_idx(self.target_episodes)
        
        # Verify that all goal indices (0-6) are available in both datasets
        for goal_idx in range(7):  # 0 to 6
            assert goal_idx in self.source_by_goal, f"Goal index {goal_idx} not found in source dataset"
            assert goal_idx in self.target_by_goal, f"Goal index {goal_idx} not found in target dataset"
        
        print(f"Loaded {len(self.source_episodes)} source episodes and {len(self.target_episodes)} target episodes")
        for goal_idx in range(7):
            print(f"Goal {goal_idx}: {len(self.source_by_goal[goal_idx])} source timesteps, "
                  f"{len(self.target_by_goal[goal_idx])} target timesteps")
    
    def _load_dataset(self, dataset_path: str, load_count: int, normalize_states: bool) -> List[Dict]:
        """
        Load episodes from dataset file.
        
        Returns:
            List of episode dictionaries, each containing:
                - obs: Observation tensor (T, obs_dim)
                - next_obs: Next observation tensor (T, obs_dim)
                - actions: Action tensor (T, act_dim)
                - dones: Done flags (T, 1)
                - rewards: Reward values (T, 1)
                - current_goal_idx: Goal indices (T,)
        """
        # Load data from h5 file
        data = h5py.File(dataset_path, "r")
        json_path = dataset_path.replace(".h5", ".json")
        json_data = load_json(json_path)
        episodes = json_data["episodes"]
        
        if load_count == -1:
            load_count = len(episodes)
        else:
            load_count = min(load_count, len(episodes))
        
        print(f"Loading {load_count} episodes from {dataset_path}")
        
        processed_episodes = []
        for eps_id in tqdm(range(load_count)):
            eps = episodes[eps_id]
            trajectory = data[f"traj_{eps['episode_id']}"]
            trajectory = load_h5_data(trajectory)
            
            # Extract data from trajectory
            observations = self._flatten_obs_dict(trajectory["obs"]["agent"])
            current_goal_idx = trajectory["obs"]["extra"]["current_goal_idx"][:-1]
            actions = trajectory["actions"]
            dones = trajectory["success"].reshape(-1, 1)
            
            # Create obs and next_obs by shifting observations
            obs = observations[:-1].copy()
            next_obs = observations[1:].copy()

            # Create dummy rewards (zeros) since they're not in the original dataset
            rewards = np.zeros_like(dones)
            
            # Convert to tensors
            obs_tensor = torch.from_numpy(obs).float().to(self.device)
            next_obs_tensor = torch.from_numpy(next_obs).float().to(self.device)
            actions_tensor = torch.from_numpy(actions).float().to(self.device)
            dones_tensor = torch.from_numpy(dones).to(self.device)
            rewards_tensor = torch.from_numpy(rewards).float().to(self.device)
            current_goal_idx_tensor = torch.from_numpy(current_goal_idx).long().to(self.device)
            
            # Store as episode dictionary
            episode = {
                "obs": obs_tensor,
                "next_obs": next_obs_tensor,
                "actions": actions_tensor,
                "dones": dones_tensor,
                "rewards": rewards_tensor,
                "current_goal_idx": current_goal_idx_tensor,
                "episode_id": eps["episode_id"]
            }
            
            processed_episodes.append(episode)
        
        # Normalize observations if requested
        if normalize_states:
            all_obs = torch.cat([ep["obs"] for ep in processed_episodes], dim=0)
            mean = all_obs.mean(dim=0, keepdim=True)
            std = all_obs.std(dim=0, keepdim=True) + 1e-6  # Add small epsilon to avoid division by zero
            
            for ep in processed_episodes:
                ep["obs"] = (ep["obs"] - mean) / std
                ep["next_obs"] = (ep["next_obs"] - mean) / std
        
        return processed_episodes
    
    def _flatten_obs_dict(self, obs_dict):
        """Flatten observation dictionary into a single array"""
        return np.concatenate([obs_dict[k] for k in obs_dict.keys()], axis=1)
    
    def _group_by_goal_idx(self, episodes: List[Dict]) -> Dict[int, List[Tuple[int, int]]]:
        """
        Group timesteps by goal index for efficient sampling.
        
        Returns:
            Dictionary mapping goal index to list of (episode_idx, timestep_idx) tuples
        """
        goal_indices = {}
        
        for ep_idx, episode in enumerate(episodes):
            current_goal_idx = episode["current_goal_idx"]
            
            for t in range(len(current_goal_idx)):
                goal_idx = current_goal_idx[t].item()
                
                if goal_idx not in goal_indices:
                    goal_indices[goal_idx] = []
                
                goal_indices[goal_idx].append((ep_idx, t))
        
        return goal_indices
    
    def _prepare_index_tensors(self):
        """
        Pre-compute tensors for efficient indexing during sampling.
        This method creates flattened tensors of episode indices, timestep indices,
        and goal indices for both source and target datasets.
        """
        # Create flattened tensors for source dataset
        source_ep_indices = []
        source_t_indices = []
        source_goal_indices = []
        
        # Also prepare data tensors for efficient batch sampling
        self._source_obs_flat = []
        self._source_next_obs_flat = []
        self._source_actions_flat = []
        self._source_dones_flat = []
        self._source_rewards_flat = []
        
        for ep_idx, episode in enumerate(self.source_episodes):
            current_goal_idx = episode["current_goal_idx"]
            num_steps = len(current_goal_idx)
            
            ep_indices = [ep_idx] * num_steps
            t_indices = list(range(num_steps))
            
            source_ep_indices.extend(ep_indices)
            source_t_indices.extend(t_indices)
            source_goal_indices.extend(current_goal_idx.tolist())
            
            # Flatten data for direct indexing
            self._source_obs_flat.append(episode["obs"])
            self._source_next_obs_flat.append(episode["next_obs"])
            self._source_actions_flat.append(episode["actions"])
            self._source_dones_flat.append(episode["dones"])
            self._source_rewards_flat.append(episode["rewards"])
        
        # Create flattened tensors for target dataset
        target_ep_indices = []
        target_t_indices = []
        target_goal_indices = []
        
        # Also prepare data tensors for efficient batch sampling
        self._target_obs_flat = []
        self._target_next_obs_flat = []
        self._target_actions_flat = []
        self._target_dones_flat = []
        self._target_rewards_flat = []
        
        for ep_idx, episode in enumerate(self.target_episodes):
            current_goal_idx = episode["current_goal_idx"]
            num_steps = len(current_goal_idx)
            
            ep_indices = [ep_idx] * num_steps
            t_indices = list(range(num_steps))
            
            target_ep_indices.extend(ep_indices)
            target_t_indices.extend(t_indices)
            target_goal_indices.extend(current_goal_idx.tolist())
            
            # Flatten data for direct indexing
            self._target_obs_flat.append(episode["obs"])
            self._target_next_obs_flat.append(episode["next_obs"])
            self._target_actions_flat.append(episode["actions"])
            self._target_dones_flat.append(episode["dones"])
            self._target_rewards_flat.append(episode["rewards"])
        
        # Convert to tensors
        self._source_ep_indices = torch.tensor(source_ep_indices, device=self.device)
        self._source_t_indices = torch.tensor(source_t_indices, device=self.device)
        self._source_goal_indices = torch.tensor(source_goal_indices, device=self.device)
        
        self._target_ep_indices = torch.tensor(target_ep_indices, device=self.device)
        self._target_t_indices = torch.tensor(target_t_indices, device=self.device)
        self._target_goal_indices = torch.tensor(target_goal_indices, device=self.device)
        
        # Concatenate data tensors for flat indexing
        self._source_obs_flat = torch.cat(self._source_obs_flat, dim=0)
        self._source_next_obs_flat = torch.cat(self._source_next_obs_flat, dim=0)
        self._source_actions_flat = torch.cat(self._source_actions_flat, dim=0)
        self._source_dones_flat = torch.cat(self._source_dones_flat, dim=0)
        self._source_rewards_flat = torch.cat(self._source_rewards_flat, dim=0)
        
        self._target_obs_flat = torch.cat(self._target_obs_flat, dim=0)
        self._target_next_obs_flat = torch.cat(self._target_next_obs_flat, dim=0)
        self._target_actions_flat = torch.cat(self._target_actions_flat, dim=0)
        self._target_dones_flat = torch.cat(self._target_dones_flat, dim=0)
        self._target_rewards_flat = torch.cat(self._target_rewards_flat, dim=0)
        
        print(f"Prepared index tensors with {len(self._source_goal_indices)} source timesteps " 
              f"and {len(self._target_goal_indices)} target timesteps")
    
    def sample_batch(self, batch_size: int, goal_idx: Optional[int] = None) -> AlignedTrajDataSample:
        """
        Sample a batch of data points with matching goal indices from both datasets.
        Uses fully vectorized operations for maximum efficiency.
        
        Args:
            batch_size: Number of samples in the batch
            goal_idx: If provided, sample only from this goal index. Otherwise, randomly select a goal index.
            
        Returns:
            AlignmentDataSample containing batched data
        """
        # If goal_idx is not provided, randomly select one
        if goal_idx is None:
            goal_idx = np.random.randint(0, 7)  # 0 to 6
        
        # Prepare index tensors if not already done
        if not hasattr(self, '_source_goal_indices'):
            self._prepare_index_tensors()
            
        # Get the indices for the specified goal_idx
        source_goal_mask = self._source_goal_indices == goal_idx
        target_goal_mask = self._target_goal_indices == goal_idx
        
        # Count how many timesteps have this goal index
        source_count = source_goal_mask.sum().item()
        target_count = target_goal_mask.sum().item()
        
        # Sample random indices for the specified goal index
        source_flat_indices = torch.nonzero(source_goal_mask, as_tuple=True)[0]
        target_flat_indices = torch.nonzero(target_goal_mask, as_tuple=True)[0]
        
        # Randomly select batch_size indices
        source_batch_indices = source_flat_indices[torch.randint(0, source_count, (batch_size,), device=self.device)]
        target_batch_indices = target_flat_indices[torch.randint(0, target_count, (batch_size,), device=self.device)]
        
        # Directly index the flattened data tensors for maximum efficiency
        source_obs = self._source_obs_flat[source_batch_indices]
        source_next_obs = self._source_next_obs_flat[source_batch_indices]
        source_actions = self._source_actions_flat[source_batch_indices]
        source_dones = self._source_dones_flat[source_batch_indices]
        source_rewards = self._source_rewards_flat[source_batch_indices]
        goal_idx_batch = self._source_goal_indices[source_batch_indices]
        
        target_obs = self._target_obs_flat[target_batch_indices]
        target_next_obs = self._target_next_obs_flat[target_batch_indices]
        target_actions = self._target_actions_flat[target_batch_indices]
        target_dones = self._target_dones_flat[target_batch_indices]
        target_rewards = self._target_rewards_flat[target_batch_indices]
        
        return AlignedTrajDataSample(
            source_obs=source_obs,
            source_next_obs=source_next_obs,
            source_actions=source_actions,
            source_dones=source_dones,
            source_rewards=source_rewards,
            target_obs=target_obs,
            target_next_obs=target_next_obs,
            target_actions=target_actions,
            target_dones=target_dones,
            target_rewards=target_rewards,
            current_goal_idx=goal_idx_batch
        )


class AlignedCalibrationTrajDataLoader:
    """
    Dataloader that yields batches of data with matching goal indices.
    """
    def __init__(self, dataset: AlignedCalibrationTrajDataset, batch_size: int, 
                 steps_per_epoch: int = 1000, shuffle_goals: bool = True):
        """
        Args:
            dataset: GoalAlignedDataset instance
            batch_size: Batch size
            steps_per_epoch: Number of batches to yield per epoch
            shuffle_goals: Whether to shuffle goal indices
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.shuffle_goals = shuffle_goals
    
    def __iter__(self):
        """Yield batches of data with matching goal indices"""
        # If shuffling, create a random order of goal indices
        if self.shuffle_goals:
            goal_indices = np.random.permutation(7)  # 0 to 6
        else:
            goal_indices = np.arange(7)
        
        # Cycle through goal indices for the requested number of steps
        for i in range(self.steps_per_epoch):
            goal_idx = goal_indices[i % 7]
            yield self.dataset.sample_batch(self.batch_size, goal_idx)
    
    def __len__(self):
        """Return the number of batches per epoch"""
        return self.steps_per_epoch


# Helper function to create a dataloader
def create_aligned_calibration_traj_dataloader(
    source_path: str, 
    target_path: str, 
    batch_size: int, 
    device: torch.device,
    load_count: int = -1, 
    steps_per_epoch: int = 1000,
    shuffle_goals: bool = True,
    normalize_states: bool = False
) -> AlignedCalibrationTrajDataLoader:
    """
    Create a dataloader that yields batches of data with matching goal indices.
    
    Args:
        source_path: Path to source dataset h5 file
        target_path: Path to target dataset h5 file
        batch_size: Batch size
        device: Device to load tensors on
        load_count: Number of episodes to load (-1 for all)
        steps_per_epoch: Number of batches to yield per epoch
        shuffle_goals: Whether to shuffle goal indices
        normalize_states: Whether to normalize observation states
        
    Returns:
        GoalAlignedDataLoader instance
    """
    dataset = AlignedCalibrationTrajDataset(
        source_path=source_path,
        target_path=target_path,
        device=device,
        load_count=load_count,
        normalize_states=normalize_states
    )
    
    dataloader = AlignedCalibrationTrajDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        shuffle_goals=shuffle_goals
    )
    
    return dataloader 