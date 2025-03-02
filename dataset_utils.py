import collections
from typing import Optional

import d4rl
import gym
import numpy as np
from tqdm import tqdm

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])


def split_into_trajectories(observations, actions, rewards, masks, dones_float,
                            next_observations):
    trajs = [[]]

    for i in tqdm(range(len(observations))):
        trajs[-1].append((observations[i], actions[i], rewards[i], masks[i],
                          dones_float[i], next_observations[i]))
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs


def merge_trajectories(trajs):
    observations = []
    actions = []
    rewards = []
    masks = []
    dones_float = []
    next_observations = []

    for traj in trajs:
        for (obs, act, rew, mask, done, next_obs) in traj:
            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            masks.append(mask)
            dones_float.append(done)
            next_observations.append(next_obs)

    return np.stack(observations), np.stack(actions), np.stack(
        rewards), np.stack(masks), np.stack(dones_float), np.stack(
            next_observations)


class Dataset(object):
    def __init__(self, observations: np.ndarray, actions: np.ndarray,
                 rewards: np.ndarray, masks: np.ndarray,
                 dones_float: np.ndarray, next_observations: np.ndarray,
                 size: int):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = size

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return Batch(observations=self.observations[indx],
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     masks=self.masks[indx],
                     next_observations=self.next_observations[indx])


class D4RLDataset(Dataset):
    def __init__(self,
                 env: gym.Env,
                 clip_to_eps: bool = True,
                 eps: float = 1e-5):
        dataset = d4rl.qlearning_dataset(env)

        if clip_to_eps:
            lim = 1 - eps
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

        dones_float = np.zeros_like(dataset['rewards'])

        for i in range(len(dones_float) - 1):
            if np.linalg.norm(dataset['observations'][i + 1] -
                              dataset['next_observations'][i]
                              ) > 1e-6 or dataset['terminals'][i] == 1.0:
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        super().__init__(dataset['observations'].astype(np.float32),
                         actions=dataset['actions'].astype(np.float32),
                         rewards=dataset['rewards'].astype(np.float32),
                         masks=1.0 - dataset['terminals'].astype(np.float32),
                         dones_float=dones_float.astype(np.float32),
                         next_observations=dataset['next_observations'].astype(
                             np.float32),
                         size=len(dataset['observations']))


class D4RLDataset_Mix(Dataset):
    def __init__(self,
                 env: gym.Env,
                 num_original: int,
                 mix_env_id: str,
                 num_mix: int,
                 ):
        dataset = D4RLDataset(env)

        mix_env = gym.make(mix_env_id)
        mix_dataset = D4RLDataset(mix_env)

        if num_mix < 0:
            num_mix= mix_dataset.size
        else:
            num_mix= min(mix_dataset.size,num_mix)

        _idxes = np.arange(0,num_mix)

        if num_original < 0:
            num_original = dataset.size
        else:
            num_original = min(dataset.size,num_original)
        _original_idxes = np.arange(0,num_original)

        observations = np.concatenate([dataset.observations[_original_idxes],mix_dataset.observations[_idxes]],axis=0)
        actions = np.concatenate([dataset.actions[_original_idxes],mix_dataset.actions[_idxes]],axis=0)
        rewards = np.concatenate([dataset.rewards[_original_idxes],mix_dataset.rewards[_idxes]],axis=0)
        masks = np.concatenate([dataset.masks[_original_idxes],mix_dataset.masks[_idxes]],axis=0)
        dones_float = np.concatenate([dataset.dones_float[_original_idxes],mix_dataset.dones_float[_idxes]],axis=0)
        next_observations = np.concatenate([dataset.next_observations[_original_idxes],mix_dataset.next_observations[_idxes]],axis=0)
        size = len(observations)

        assert size == num_original + num_mix

        return super().__init__(
            observations,
            actions,
            rewards,
            masks,
            dones_float,
            next_observations,
            size)

class RobomimicDataset(Dataset):
    def __init__(self,
                 env: gym.Env):
        trajs = env.get_dataset(ignore_done=False, truncate_if_done=False, shape_reward=True)

        obs, acs, rs, dones, next_obs = [], [], [], [], []

        for traj in trajs:
            obs.append(traj.states[:-1])
            acs.append(traj.actions)
            rs.append(traj.rewards)
            dones.append(traj.dones)
            next_obs.append(traj.states[1:])

        obs = np.concatenate(obs,axis=0)
        acs = np.concatenate(acs,axis=0)
        rs = np.concatenate(rs,axis=0)
        dones = np.concatenate(dones,axis=0)
        next_obs = np.concatenate(next_obs,axis=0)

        super().__init__(observations=obs.astype(np.float32),
                         actions=acs.astype(np.float32),
                         rewards=rs.astype(np.float32),
                         masks=1.0 - dones.astype(np.float32),
                         dones_float=dones.astype(np.float32),
                         next_observations=next_obs.astype(
                             np.float32),
                         size=len(obs))

class ReplayBuffer(Dataset):
    def __init__(self, observation_space: gym.spaces.Box, action_dim: int,
                 capacity: int):

        observations = np.empty((capacity, *observation_space.shape),
                                dtype=observation_space.dtype)
        actions = np.empty((capacity, action_dim), dtype=np.float32)
        rewards = np.empty((capacity, ), dtype=np.float32)
        masks = np.empty((capacity, ), dtype=np.float32)
        dones_float = np.empty((capacity, ), dtype=np.float32)
        next_observations = np.empty((capacity, *observation_space.shape),
                                     dtype=observation_space.dtype)
        super().__init__(observations=observations,
                         actions=actions,
                         rewards=rewards,
                         masks=masks,
                         dones_float=dones_float,
                         next_observations=next_observations,
                         size=0)

        self.size = 0

        self.insert_index = 0
        self.capacity = capacity

    def initialize_with_dataset(self, dataset: Dataset,
                                num_samples: Optional[int]):
        assert self.insert_index == 0, 'Can insert a batch online in an empty replay buffer.'

        dataset_size = len(dataset.observations)

        if num_samples is None:
            num_samples = dataset_size
        else:
            num_samples = min(dataset_size, num_samples)
        assert self.capacity >= num_samples, 'Dataset cannot be larger than the replay buffer capacity.'

        if num_samples < dataset_size:
            perm = np.random.permutation(dataset_size)
            indices = perm[:num_samples]
        else:
            indices = np.arange(num_samples)

        self.observations[:num_samples] = dataset.observations[indices]
        self.actions[:num_samples] = dataset.actions[indices]
        self.rewards[:num_samples] = dataset.rewards[indices]
        self.masks[:num_samples] = dataset.masks[indices]
        self.dones_float[:num_samples] = dataset.dones_float[indices]
        self.next_observations[:num_samples] = dataset.next_observations[
            indices]

        self.insert_index = num_samples
        self.size = num_samples

    def insert(self, observation: np.ndarray, action: np.ndarray,
               reward: float, mask: float, done_float: float,
               next_observation: np.ndarray):
        self.observations[self.insert_index] = observation
        self.actions[self.insert_index] = action
        self.rewards[self.insert_index] = reward
        self.masks[self.insert_index] = mask
        self.dones_float[self.insert_index] = done_float
        self.next_observations[self.insert_index] = next_observation

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
