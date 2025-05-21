import os
import uuid
from typing import Union, List

import h5py
import numpy as np
import torch
from h5py import File, Dataset, VirtualSource, VirtualLayout
from torch.utils.data import Dataset as TorchDataset
from tqdm.auto import tqdm

from mani_skill.utils import common
from mani_skill.utils.io_utils import load_json

# loads h5 data into memory for faster access
def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out


class ManiSkillTrajectoryDataset(TorchDataset):
    """
    A general torch Dataset you can drop in and use immediately with just about any trajectory .h5 data generated from ManiSkill.
    This class simply is a simple starter code to load trajectory data easily, but does not do any data transformation or anything
    advanced. We recommend you to copy this code directly and modify it for more advanced use cases

    Args:
        dataset_file (str): path to the .h5 file containing the data you want to load
        load_count (int): the number of trajectories from the dataset to load into memory. If -1, will load all into memory
        success_only (bool): whether to skip trajectories that are not successful in the end. Default is false
        device: The location to save data to. If None will store as numpy (the default), otherwise will move data to that device
    """

    def __init__(
        self, dataset_file: str, load_count=-1, success_only: bool = False, device=None
    ) -> None:
        self.dataset_file = dataset_file
        self.device = device
        self.data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        self.json_data = load_json(json_path)
        self.episodes = self.json_data["episodes"]
        self.env_info = self.json_data["env_info"]
        self.env_id = self.env_info["env_id"]
        self.env_kwargs = self.env_info["env_kwargs"]

        self.obs = None
        self.actions = []
        self.terminated = []
        self.truncated = []
        self.success, self.fail, self.rewards = None, None, None
        if load_count == -1:
            load_count = len(self.episodes)
        for eps_id in tqdm(range(load_count)):
            eps = self.episodes[eps_id]
            if success_only:
                assert (
                    "success" in eps
                ), "episodes in this dataset do not have the success attribute, cannot load dataset with success_only=True"
                if not eps["success"]:
                    continue
            trajectory = self.data[f"traj_{eps['episode_id']}"]
            trajectory = load_h5_data(trajectory)
            eps_len = len(trajectory["actions"])

            # exclude the final observation as most learning workflows do not use it
            obs = common.index_dict_array(trajectory["obs"], slice(eps_len))
            if eps_id == 0:
                self.obs = obs
            else:
                self.obs = common.append_dict_array(self.obs, obs)

            self.actions.append(trajectory["actions"])
            self.terminated.append(trajectory["terminated"])
            self.truncated.append(trajectory["truncated"])

            # handle data that might optionally be in the trajectory
            if "rewards" in trajectory:
                if self.rewards is None:
                    self.rewards = [trajectory["rewards"]]
                else:
                    self.rewards.append(trajectory["rewards"])
            if "success" in trajectory:
                if self.success is None:
                    self.success = [trajectory["success"]]
                else:
                    self.success.append(trajectory["success"])
            if "fail" in trajectory:
                if self.fail is None:
                    self.fail = [trajectory["fail"]]
                else:
                    self.fail.append(trajectory["fail"])

        self.actions = np.vstack(self.actions)
        self.terminated = np.concatenate(self.terminated)
        self.truncated = np.concatenate(self.truncated)

        if self.rewards is not None:
            self.rewards = np.concatenate(self.rewards)
        if self.success is not None:
            self.success = np.concatenate(self.success)
        if self.fail is not None:
            self.fail = np.concatenate(self.fail)

        def remove_np_uint16(x: Union[np.ndarray, dict]):
            if isinstance(x, dict):
                for k in x.keys():
                    x[k] = remove_np_uint16(x[k])
                return x
            else:
                if x.dtype == np.uint16:
                    return x.astype(np.int32)
                return x

        # uint16 dtype is used to conserve disk space and memory
        # you can optimize this dataset code to keep it as uint16 and process that
        # dtype of data yourself. for simplicity we simply cast to a int32 so
        # it can automatically be converted to torch tensors without complaint
        self.obs = remove_np_uint16(self.obs)

        if device is not None:
            self.actions = common.to_tensor(self.actions, device=device)
            self.obs = common.to_tensor(self.obs, device=device)
            self.terminated = common.to_tensor(self.terminated, device=device)
            self.truncated = common.to_tensor(self.truncated, device=device)
            if self.rewards is not None:
                self.rewards = common.to_tensor(self.rewards, device=device)
            if self.success is not None:
                self.success = common.to_tensor(self.terminated, device=device)
            if self.fail is not None:
                self.fail = common.to_tensor(self.truncated, device=device)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        action = common.to_tensor(self.actions[idx], device=self.device)
        obs = common.index_dict_array(self.obs, idx, inplace=False)

        res = dict(
            obs=obs,
            action=action,
            terminated=self.terminated[idx],
            truncated=self.truncated[idx],
        )
        if self.rewards is not None:
            res.update(reward=self.rewards[idx])
        if self.success is not None:
            res.update(success=self.success[idx])
        if self.fail is not None:
            res.update(fail=self.fail[idx])
        return res



def load_h5_data_to_virtual(file_path, h5data, tree_path=[]):
    out = dict()
    for k in h5data.keys():
        if isinstance(h5data[k], Dataset):
            name = "/".join(tree_path + [k])
            out[k] = VirtualSource(file_path, name, shape=h5data[k].shape, dtype=h5data[k].dtype)
        else:
            out[k] = load_h5_data_to_virtual(file_path, h5data[k], tree_path=tree_path + [k])
    return out


class ManiSkillMemEffDataset(TorchDataset):
    def __init__(self, dataset_file: str, device: torch.device, load_count) -> None:
        self.dataset_file = dataset_file

        json_path = dataset_file.replace(".h5", ".json")
        self.json_data = load_json(json_path)
        self.episodes = self.json_data["episodes"]

        self.env_info = self.json_data["env_info"]
        self.env_id = self.env_info["env_id"]
        self.env_kwargs = self.env_info["env_kwargs"]

        self.rgb = []
        self.depth = []
        self.actions = []
        self.dones = []
        self.states = []
        self.total_frames = 0
        self.device = device
        if load_count is None:
            load_count = len(self.episodes)

        self.files_to_close: List[File] = []

        states_dsets = []
        actions_vsources = []
        rgb_vsources = []
        depth_vsources = []
        with File(self.dataset_file, "r") as data:
            for eps_id in tqdm(range(load_count)):
                eps = self.episodes[eps_id]
                traj_name = f"traj_{eps['episode_id']}"
                trajectory = data[traj_name]
                vstrajectory = load_h5_data_to_virtual(self.dataset_file, trajectory, tree_path=[traj_name])
                agent = vstrajectory["obs"]["agent"]
                extra = vstrajectory["obs"]["extra"]

                agent_state = self.flatten_state_dict_with_space(agent)
                extra_state = self.flatten_state_dict_with_space(extra)
                
                state = self.concat_virtual_datasets([agent_state, extra_state], axis=1)
                
                states_dsets.append(state)
                actions_vsources.append(vstrajectory["actions"])

                # we use :-1 here to ignore the last observation as that
                # is the terminal observation which has no actions
                camera_name = "base_camera"
                if camera_name not in vstrajectory["obs"]["sensor_data"]:
                    camera_name = list(vstrajectory["obs"]["sensor_data"].keys())[0]
                rgb_vsources.append(vstrajectory["obs"]["sensor_data"][camera_name]["rgb"][:-1])
                depth_vsources.append(vstrajectory["obs"]["sensor_data"][camera_name]["depth"][:-1])

        self.states = self.concat_virtual_datasets(states_dsets, axis=0)
        self.actions = self.concat_vsources(actions_vsources, axis=0)

        # TODO:Remember to divide by 255.0 and 1024.0 respectively when converting to numpy
        self.rgb = self.concat_vsources(rgb_vsources, axis=0)
        self.depth = self.concat_vsources(depth_vsources, axis=0)

        assert self.depth.shape[0] == self.actions.shape[0]
        assert self.rgb.shape[0] == self.actions.shape[0]


    def flatten_state_dict_with_space(self, state_dict: dict) -> Dataset:
        ep_state_dsets = []
        for key in state_dict.keys():
            value = state_dict[key]
            assert isinstance(value, VirtualSource), "state_dict should only contain VirtualSource objects"
            if len(value.shape) > 2:
                raise AssertionError(
                    "The dimension of {} should not be more than 2.".format(key)
                )
            if len(value.shape) == 1:
                value = self.add_axis(value, 1)
            
            if isinstance(value, VirtualSource):
                value = self.virtual_source_to_dataset(value)
            ep_state_dsets.append(value)
        if len(ep_state_dsets) == 0:
            raise ValueError("Cannot flatten an empty state_dict")
        else:
            return self.concat_virtual_datasets(ep_state_dsets, axis=1)
        

    def virtual_source_to_dataset(self, vsource: VirtualSource) -> Dataset:
        """
        Converts an h5py VirtualSource object to a Dataset object.
        
        Args:
            vsource: An h5py VirtualSource object
            
        Returns:
            A Dataset object
        """

        layout = VirtualLayout(shape=vsource.shape, dtype=vsource.dtype)
        layout[:] = vsource

        unique_id = str(uuid.uuid4())[:8]
        filename = f"temp_vsource_direct_{unique_id}.h5"
        temp_file = File(filename, 'w', driver='core', backing_store=False)
        vdset = temp_file.create_virtual_dataset('vdset', layout)

        self.files_to_close.append(temp_file)
        return vdset


    def add_axis(self, vsource: VirtualSource, axis: int) -> Dataset:
        """
        Adds a dimension to an HDF5 dataset at the specified axis using virtual datasets.
        
        Args:
            vsource: An h5py VirtualSource object
            axis: The axis position to insert the new dimension
            close_file: If True, closes the temporary file and returns only the data
            
        Returns:
            Expanded virtual dataset
        """
        # Generate a unique ID for the temp file
        unique_id = str(uuid.uuid4())[:8]
        
        dset_shape = list(vsource.shape)
        dset_shape.insert(axis, 1)
        new_layout = VirtualLayout(shape=tuple(dset_shape), dtype=vsource.dtype)
        
        new_layout[:] = vsource
        
        # Create in-memory file with the virtual dataset
        filename = f"temp_axis_{unique_id}.h5"
        temp_file = File(filename, 'w', driver='core', backing_store=False)
        vdset = temp_file.create_virtual_dataset('vdset', new_layout)
        
        self.files_to_close.append(temp_file)
        return vdset
    

    def concat_vsources(self, virtual_sources: List[VirtualSource], axis: int = 0) -> Dataset:
        """
        Concatenates multiple HDF5 virtual sources along the specified axis.
        
        Args:
            virtual_sources: List of h5py VirtualSource objects to concatenate
            axis: Axis along which to concatenate (default: 0)
            close_file: If True, closes the temporary file and returns only the data
            
        Returns:
            Merged virtual dataset
            
        Raises:
            ValueError: If datasets have incompatible shapes or dtypes, or if the list is empty
        """        
        # Input validation
        if not virtual_sources:
            raise ValueError("Cannot concatenate an empty list of datasets")
        
        if len(virtual_sources) == 1:
            return self.virtual_source_to_dataset(virtual_sources[0])
        
        # Reference vsource (first)
        ref_vsource = virtual_sources[0]
        ref_dtype = ref_vsource.dtype
        ref_shape = list(ref_vsource.shape)
        ref_ndim = len(ref_shape)
        
        # Validate all datasets for compatibility
        total_size_along_axis = ref_shape[axis]
        for i, vsource in enumerate(virtual_sources[1:], 1):
            # Check dtype
            if vsource.dtype != ref_dtype:
                raise ValueError(f"Virtual Source {i} has incompatible dtype: {vsource.dtype} vs {ref_dtype}")
            
            # Check dimensions
            vsource_shape = list(vsource.shape)
            if len(vsource_shape) != ref_ndim:
                raise ValueError(f"Virtual Source {i} has different dimensions: {len(vsource_shape)} vs {ref_ndim}")
            
            # Check shape compatibility (all dimensions except concat axis must match)
            for dim in range(ref_ndim):
                if dim != axis and vsource_shape[dim] != ref_shape[dim]:
                    raise ValueError(f"Virtual Source {i} has dimension mismatch at axis {dim}: {vsource_shape[dim]} vs {ref_shape[dim]}")
            
            # Accumulate size along concat axis
            total_size_along_axis += vsource_shape[axis]
        
        # Generate a unique ID for the temp file
        unique_id = str(uuid.uuid4())[:8]
        
        # Create the layout with the combined shape
        new_shape = list(ref_shape)
        new_shape[axis] = total_size_along_axis
        layout = VirtualLayout(shape=tuple(new_shape), dtype=ref_dtype)
        
        # Map each source dataset to its respective position
        current_pos = 0
        for i, vsource in enumerate(virtual_sources):
            # Create slice for this dataset
            slices = [slice(None)] * ref_ndim
            size_along_axis = vsource.shape[axis]
            slices[axis] = slice(current_pos, current_pos + size_along_axis)
            
            # Map this source to the layout
            layout[tuple(slices)] = vsource
            
            # Update position for next dataset
            current_pos += size_along_axis
        
        # Create the virtual dataset in a memory-only file
        filename = f"temp_multiconcat_{unique_id}.h5"
        temp_file = File(filename, 'w', driver='core', backing_store=False)
        merged_dset = temp_file.create_virtual_dataset('merged_vdset', layout)
    
        self.files_to_close.append(temp_file)
        return merged_dset
    

    def concat_virtual_datasets(self, datasets: List[Dataset], axis: int = 0) -> Dataset:
        """
        Concatenates multiple HDF5 datasets along the specified axis.
        
        Args:
            datasets: List of h5py Dataset objects to concatenate
            axis: Axis along which to concatenate (default: 0)
            
        Returns:
            A Dataset object with concatenated data
        """
        # Input validation
        if not datasets:
            raise ValueError("Cannot concatenate an empty list of datasets")
        
        if len(datasets) == 1:
            return datasets[0]
        
        # Reference dataset (first)
        ref_dataset = datasets[0]
        ref_dtype = ref_dataset.dtype
        ref_shape = list(ref_dataset.shape)
        ref_ndim = len(ref_shape)
        
        # Validate all datasets for compatibility
        total_size_along_axis = ref_shape[axis]
        for i, dset in enumerate(datasets[1:], 1):
            # Check dtype
            if dset.dtype != ref_dtype:
                raise ValueError(f"Dataset {i} has incompatible dtype: {dset.dtype} vs {ref_dtype}")
            
            # Check dimensions
            dset_shape = list(dset.shape)
            if len(dset_shape) != ref_ndim:
                raise ValueError(f"Dataset {i} has different dimensions: {len(dset_shape)} vs {ref_ndim}")
            
            # Check shape compatibility (all dimensions except concat axis must match)
            for dim in range(ref_ndim):
                if dim != axis and dset_shape[dim] != ref_shape[dim]:
                    raise ValueError(f"Dataset {i} has dimension mismatch at axis {dim}: {dset_shape[dim]} vs {ref_shape[dim]}")
            
            # Accumulate size along concat axis
            total_size_along_axis += dset_shape[axis]
        
        # Generate a unique ID for the temp file
        unique_id = str(uuid.uuid4())[:8]
        
        # Create a temporary file to store the concatenated data
        filename = f"temp_dset_concat_{unique_id}.h5"
        temp_file = File(filename, 'w', driver='core', backing_store=False)
        
        # Create the output dataset in the temporary file
        new_shape = list(ref_shape)
        new_shape[axis] = total_size_along_axis
        output_dset = temp_file.create_dataset('concat_data', shape=tuple(new_shape), dtype=ref_dtype)
        
        # Copy data from each input dataset to the output dataset
        current_pos = 0
        for dset in datasets:
            # Create slice for this dataset
            slices = [slice(None)] * ref_ndim
            size_along_axis = dset.shape[axis]
            slices[axis] = slice(current_pos, current_pos + size_along_axis)
            
            # Copy data from this dataset to the output dataset
            output_dset[tuple(slices)] = dset[:]
            
            # Update position for next dataset
            current_pos += size_along_axis
        
        self.files_to_close.append(temp_file)
        return output_dset


    def close(self):
        for f in self.files_to_close:
            fname = f.filename
            f.close()
            os.remove(fname)


    def __del__(self):
        self.close()


    def __len__(self):
        return len(self.rgb)


    def __getitem__(self, idx):
        out = {}
        out["state"] = torch.from_numpy(self.states[idx]).float().to(device=self.device)
        out["action"] = torch.from_numpy(self.actions[idx]).float().to(device=self.device)
        depth = torch.from_numpy(self.depth[idx]).float().to(device=self.device)
        rgb = torch.from_numpy(self.rgb[idx]).float().to(device=self.device)
        out["rgbd"] = torch.cat([rgb, depth], dim=-1)
        
        # Handle states that might be virtual datasets
        return out