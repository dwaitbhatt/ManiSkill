import torch
from aligned_calibration_dataset import create_aligned_calibration_traj_dataloader

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataloader using the helper function
    dataloader = create_aligned_calibration_traj_dataloader(
        source_path="/home/dwait/ManiSkill/demos/XembCalibration-v1/motionplanning/panda_calibration_traj_n10.state_dict.pd_joint_vel.physx_cpu.h5",
        target_path="/home/dwait/ManiSkill/demos/XembCalibration-v1/motionplanning/xarm6_calibration_traj_n10.state_dict.pd_joint_vel.physx_cpu.h5",
        batch_size=32,
        device=device,
        load_count=10,  # Load 100 episodes for example
        steps_per_epoch=1000,  # Number of batches per epoch
        shuffle_goals=False,
        normalize_states=False
    )
    
    print(f"Total batches per epoch: {len(dataloader)}")
    
    # Example training loop
    for batch_idx, batch in enumerate(dataloader):
        # Access the data
        source_obs = batch.source_obs
        source_next_obs = batch.source_next_obs
        source_actions = batch.source_actions
        source_dones = batch.source_dones
        source_rewards = batch.source_rewards
        
        target_obs = batch.target_obs
        target_next_obs = batch.target_next_obs
        target_actions = batch.target_actions
        target_dones = batch.target_dones
        target_rewards = batch.target_rewards
        
        goal_indices = batch.current_goal_idx
        
        # Print shapes and goal indices for verification
        print(f"Batch {batch_idx}:")
        print(f"Source obs shape: {source_obs.shape}")
        print(f"Target obs shape: {target_obs.shape}")
        print(f"Goal indices unique values: {goal_indices.unique().tolist()}")  # Should only have one unique value per batch
        
        # Convert to standard ReplayBufferSample if needed for existing code
        source_replay_sample = batch.to_source_replay_buffer_sample()
        target_replay_sample = batch.to_target_replay_buffer_sample()
        
        # Use these samples with your existing training code
        # train_step(source_replay_sample, target_replay_sample, goal_indices)
        
        if batch_idx >= 2:  # Just show a few batches for the example
            break

if __name__ == "__main__":
    main() 