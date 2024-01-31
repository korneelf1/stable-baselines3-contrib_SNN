from functools import partial
from typing import Callable, Generator, Optional, Tuple, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.vec_env import VecNormalize

from sb3_contrib.common.spiking.type_aliases import (
    RecurrentDictRolloutBufferSamples,
    RecurrentRolloutBufferSamples,
    SNNStates,
)


def pad(
    seq_start_indices: np.ndarray,
    seq_end_indices: np.ndarray,
    device: th.device,
    tensor: np.ndarray,
    padding_value: float = 0.0,
) -> th.Tensor:
    """
    Chunk sequences and pad them to have constant dimensions.

    :param seq_start_indices: Indices of the transitions that start a sequence
    :param seq_end_indices: Indices of the transitions that end a sequence
    :param device: PyTorch device
    :param tensor: Tensor of shape (batch_size, *tensor_shape)
    :param padding_value: Value used to pad sequence to the same length
        (zero padding by default)
    :return: (n_seq, max_length, *tensor_shape)
    """
    # Create sequences given start and end
    seq = [th.tensor(tensor[start : end + 1], device=device) for start, end in zip(seq_start_indices, seq_end_indices)]
    return th.nn.utils.rnn.pad_sequence(seq, batch_first=True, padding_value=padding_value)


def pad_and_flatten(
    seq_start_indices: np.ndarray,
    seq_end_indices: np.ndarray,
    device: th.device,
    tensor: np.ndarray,
    padding_value: float = 0.0,
) -> th.Tensor:
    """
    Pad and flatten the sequences of scalar values,
    while keeping the sequence order.
    From (batch_size, 1) to (n_seq, max_length, 1) -> (n_seq * max_length,)

    :param seq_start_indices: Indices of the transitions that start a sequence
    :param seq_end_indices: Indices of the transitions that end a sequence
    :param device: PyTorch device (cpu, gpu, ...)
    :param tensor: Tensor of shape (max_length, n_seq, 1)
    :param padding_value: Value used to pad sequence to the same length
        (zero padding by default)
    :return: (n_seq * max_length,) aka (padded_batch_size,)
    """
    return pad(seq_start_indices, seq_end_indices, device, tensor, padding_value).flatten()


def create_sequencers(
    episode_starts: np.ndarray,
    env_change: np.ndarray,
    device: th.device,
) -> Tuple[np.ndarray, Callable, Callable]:
    """
    Create the utility function to chunk data into
    sequences and pad them to create fixed size tensors.

    :param episode_starts: Indices where an episode starts
    :param env_change: Indices where the data collected
        come from a different env (when using multiple env for data collection)
    :param device: PyTorch device
    :return: Indices of the transitions that start a sequence,
        pad and pad_and_flatten utilities tailored for this batch
        (sequence starts and ends indices are fixed)
    """
    # Create sequence if env changes too
    seq_start = np.logical_or(episode_starts, env_change).flatten()
    # First index is always the beginning of a sequence
    seq_start[0] = True
    # Retrieve indices of sequence starts
    seq_start_indices = np.where(seq_start == True)[0]  # noqa: E712
    # End of sequence are just before sequence starts
    # Last index is also always end of a sequence
    seq_end_indices = np.concatenate([(seq_start_indices - 1)[1:], np.array([len(episode_starts)])])

    # Create padding method for this minibatch
    # to avoid repeating arguments (seq_start_indices, seq_end_indices)
    local_pad = partial(pad, seq_start_indices, seq_end_indices, device)
    local_pad_and_flatten = partial(pad_and_flatten, seq_start_indices, seq_end_indices, device)
    return seq_start_indices, local_pad, local_pad_and_flatten


class RecurrentRolloutBuffer(RolloutBuffer):
    """
    Rollout buffer that also stores the SNN cell and hidden states.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param hidden_state_shape: Shape of the buffer that will collect snn states
        (n_steps, snn.num_layers, n_envs, snn.hidden_size)
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        hidden_state_shape: Tuple[int, int, int, int],
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        self.hidden_state_shape = hidden_state_shape
        self.seq_start_indices, self.seq_end_indices = None, None
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs)
        print('Recurrent Rollout Buffer generated.')
        
    def reset(self):
        super().reset()
        self.neuron_states_pi = np.zeros(self.hidden_state_shape, dtype=np.float32)
        self.neuron_states_vf = np.zeros(self.hidden_state_shape, dtype=np.float32)
        neuron_states_pi_extractor_shape = (self.hidden_state_shape[0], 1, self.hidden_state_shape[2], self.action_space.n)
        neuron_states_vf_extractor_shape = (self.hidden_state_shape[0], 1, self.hidden_state_shape[2], 1)
        self.neuron_states_pi_extractor = np.zeros(neuron_states_pi_extractor_shape, dtype=np.float32)
        self.neuron_states_vf_extractor = np.zeros(neuron_states_vf_extractor_shape, dtype=np.float32)

    def add(self, *args, snn_states: SNNStates, **kwargs) -> None:
        """
        :param hidden_states: SNN potential states and MLP extractor states
        """
        self.neuron_states_pi[self.pos] = np.array(snn_states.pi_potentials.cpu().numpy())
        self.neuron_states_vf[self.pos] = np.array(snn_states.vf_potentials.cpu().numpy())
        self.neuron_states_pi_extractor[self.pos] = np.array(snn_states.pi_extractor.cpu().numpy())
        self.neuron_states_vf_extractor[self.pos] = np.array(snn_states.vf_extractor.cpu().numpy())

        super().add(*args, **kwargs)

    def get(self, batch_size: Optional[int] = None) -> Generator[RecurrentRolloutBufferSamples, None, None]:
        assert self.full, "Rollout buffer must be full before sampling from it"

        # Prepare the data
        if not self.generator_ready:
            # hidden_state_shape = (self.n_steps, snn.num_layers, self.n_envs, snn.hidden_size)
            # swap first to (self.n_steps, self.n_envs, snn.num_layers, snn.hidden_size)
            for tensor in ["neuron_states_pi", "neuron_states_vf", "neuron_states_pi_extractor", "neuron_states_vf_extractor"]:
                self.__dict__[tensor] = self.__dict__[tensor].swapaxes(1, 2)

            # flatten but keep the sequence order
            # 1. (n_steps, n_envs, *tensor_shape) -> (n_envs, n_steps, *tensor_shape)
            # 2. (n_envs, n_steps, *tensor_shape) -> (n_envs * n_steps, *tensor_shape)
            for tensor in [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "neuron_states_pi",
                "neuron_states_vf",
                "neuron_states_pi_extractor",
                "neuron_states_vf_extractor",
                "episode_starts",
            ]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        # Sampling strategy that allows any mini batch size but requires
        # more complexity and use of padding
        # Trick to shuffle a bit: keep the sequence order
        # but split the indices in two
        split_index = np.random.randint(self.buffer_size * self.n_envs)
        indices = np.arange(self.buffer_size * self.n_envs)
        indices = np.concatenate((indices[split_index:], indices[:split_index]))

        env_change = np.zeros(self.buffer_size * self.n_envs).reshape(self.buffer_size, self.n_envs)
        # Flag first timestep as change of environment
        env_change[0, :] = 1.0
        env_change = self.swap_and_flatten(env_change)

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            batch_inds = indices[start_idx : start_idx + batch_size]
            yield self._get_samples(batch_inds, env_change)
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env_change: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> RecurrentRolloutBufferSamples:
        # Retrieve sequence starts and utility function
        self.seq_start_indices, self.pad, self.pad_and_flatten = create_sequencers(
            self.episode_starts[batch_inds], env_change[batch_inds], self.device
        )

        # Number of sequences
        n_seq = len(self.seq_start_indices)
        max_length = self.pad(self.actions[batch_inds]).shape[1]
        padded_batch_size = n_seq * max_length
        # We retrieve the snn hidden states that will allow
        # to properly initialize the SNN at the beginning of each sequence
        snn_states_pi = (
            # 1. (n_envs * n_steps, n_layers, dim) -> (batch_size, n_layers, dim)
            # 2. (batch_size, n_layers, dim)  -> (n_seq, n_layers, dim)
            # 3. (n_seq, n_layers, dim) -> (n_layers, n_seq, dim)
            self.neuron_states_pi[batch_inds][self.seq_start_indices].swapaxes(0, 1),
        )
        snn_extractor_pi = (
            # 1. (n_envs * n_steps, n_layers, dim) -> (batch_size, n_layers, dim)
            # 2. (batch_size, n_layers, dim)  -> (n_seq, n_layers, dim)
            # 3. (n_seq, n_layers, dim) -> (n_layers, n_seq, dim)
            self.neuron_states_pi_extractor[batch_inds][self.seq_start_indices].swapaxes(0, 1),
        )
        snn_states_vf = (
            # (n_envs * n_steps, n_layers, dim) -> (n_layers, n_seq, dim)
            self.neuron_states_vf[batch_inds][self.seq_start_indices].swapaxes(0, 1),
        )
        snn_extractor_vf = (
            # (n_envs * n_steps, n_layers, dim) -> (n_layers, n_seq, dim)
            self.neuron_states_vf_extractor[batch_inds][self.seq_start_indices].swapaxes(0, 1),
        )

        snn_states_pi = self.to_torch(snn_states_pi[0]).contiguous()
        snn_states_vf = self.to_torch(snn_states_vf[0]).contiguous()
        snn_extractor_pi = self.to_torch(snn_extractor_pi[0]).contiguous()
        snn_extractor_vf = self.to_torch(snn_extractor_vf[0]).contiguous()

        return RecurrentRolloutBufferSamples(
            # (batch_size, obs_dim) -> (n_seq, max_length, obs_dim) -> (n_seq * max_length, obs_dim)
            observations=self.pad(self.observations[batch_inds]).reshape((padded_batch_size, *self.obs_shape)),
            actions=self.pad(self.actions[batch_inds]).reshape((padded_batch_size,) + self.actions.shape[1:]),
            old_values=self.pad_and_flatten(self.values[batch_inds]),
            old_log_prob=self.pad_and_flatten(self.log_probs[batch_inds]),
            advantages=self.pad_and_flatten(self.advantages[batch_inds]),
            returns=self.pad_and_flatten(self.returns[batch_inds]),
            snn_states=SNNStates(pi_potentials=snn_states_pi, vf_potentials=snn_states_vf, pi_extractor=snn_extractor_pi, vf_extractor=snn_extractor_vf),
            episode_starts=self.pad_and_flatten(self.episode_starts[batch_inds]),
            mask=self.pad_and_flatten(np.ones_like(self.returns[batch_inds])),
        )


class RecurrentDictRolloutBuffer(DictRolloutBuffer):
    """
    Dict Rollout buffer used in on-policy algorithms like A2C/PPO.
    Extends the RecurrentRolloutBuffer to use dictionary observations

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param hidden_state_shape: Shape of the buffer that will collect snn states
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        hidden_state_shape: Tuple[int, int, int, int],
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        self.hidden_state_shape = hidden_state_shape
        self.seq_start_indices, self.seq_end_indices = None, None
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs=n_envs)
        print('Recurrent Dict Rollout Buffer generated.')

    def reset(self):
        super().reset()
        self.neuron_states_pi = np.zeros(self.hidden_state_shape, dtype=np.float32)
        self.neuron_states_pi_extractor = np.zeros(self.hidden_state_shape, dtype=np.float32)
        neuron_states_pi_extractor_shape = (self.hidden_state_shape[0], self.hidden_state_shape[1], self.hidden_state_shape[2], self.action_space.n)
        neuron_states_vf_extractor_shape = (self.hidden_state_shape[0], self.hidden_state_shape[1], self.hidden_state_shape[2], 1)
        self.neuron_states_pi_extractor = np.zeros(neuron_states_pi_extractor_shape, dtype=np.float32)
        self.neuron_states_vf_extractor = np.zeros(neuron_states_vf_extractor_shape, dtype=np.float32)

    def add(self, *args, snn_states: SNNStates, **kwargs) -> None:
        """
        :param hidden_states: SNN cell and hidden state
        """
        self.neuron_states_pi[self.pos] = np.array(snn_states.pi_potentials.cpu().numpy())
        self.neuron_states_vf[self.pos] = np.array(snn_states.vf_potentials.cpu().numpy())
        self.neuron_states_pi_extractor[self.pos] = np.array(snn_states.pi_extractor.cpu().numpy())
        self.neuron_states_vf_extractor[self.pos] = np.array(snn_states.vf_extractor.cpu().numpy())

        super().add(*args, **kwargs)

    def get(self, batch_size: Optional[int] = None) -> Generator[RecurrentDictRolloutBufferSamples, None, None]:
        assert self.full, "Rollout buffer must be full before sampling from it"

        # Prepare the data
        if not self.generator_ready:
            # hidden_state_shape = (self.n_steps, snn.num_layers, self.n_envs, snn.hidden_size)
            # swap first to (self.n_steps, self.n_envs, snn.num_layers, snn.hidden_size)
            for tensor in ["neuron_states_pi", "neuron_states_vf", "neuron_states_pi_extractor", "neuron_states_vf_extractor"]:
                self.__dict__[tensor] = self.__dict__[tensor].swapaxes(1, 2)

            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)

            for tensor in [
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "neuron_states_pi",
                "neuron_states_vf",
                "neuron_states_pi_extractor",
                "neuron_states_vf_extractor",
                "episode_starts",
            ]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        # Trick to shuffle a bit: keep the sequence order
        # but split the indices in two
        split_index = np.random.randint(self.buffer_size * self.n_envs)
        indices = np.arange(self.buffer_size * self.n_envs)
        indices = np.concatenate((indices[split_index:], indices[:split_index]))

        env_change = np.zeros(self.buffer_size * self.n_envs).reshape(self.buffer_size, self.n_envs)
        # Flag first timestep as change of environment
        env_change[0, :] = 1.0
        env_change = self.swap_and_flatten(env_change)

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            batch_inds = indices[start_idx : start_idx + batch_size]
            yield self._get_samples(batch_inds, env_change)
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env_change: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> RecurrentDictRolloutBufferSamples:
        # Retrieve sequence starts and utility function
        self.seq_start_indices, self.pad, self.pad_and_flatten = create_sequencers(
            self.episode_starts[batch_inds], env_change[batch_inds], self.device
        )

        n_seq = len(self.seq_start_indices)
        max_length = self.pad(self.actions[batch_inds]).shape[1]
        padded_batch_size = n_seq * max_length
        # We retrieve the snn hidden states that will allow
        # to properly initialize the SNN at the beginning of each sequence
        snn_states_pi = (
            # (n_envs * n_steps, n_layers, dim) -> (n_layers, n_seq, dim)
            self.neuron_states_pi[batch_inds][self.seq_start_indices].swapaxes(0, 1),
        )
        snn_extractor_pi = (
            # 1. (n_envs * n_steps, n_layers, dim) -> (batch_size, n_layers, dim)
            # 2. (batch_size, n_layers, dim)  -> (n_seq, n_layers, dim)
            # 3. (n_seq, n_layers, dim) -> (n_layers, n_seq, dim)
            self.neuron_states_pi_extractor[batch_inds][self.seq_start_indices].swapaxes(0, 1),
        )

        snn_states_vf = (
            # (n_envs * n_steps, n_layers, dim) -> (n_layers, n_seq, dim)
            self.neuron_states_vf[batch_inds][self.seq_start_indices].swapaxes(0, 1),
        )
        snn_extractor_vf = (
            # (n_envs * n_steps, n_layers, dim) -> (n_layers, n_seq, dim)
            self.neuron_states_vf_extractor[batch_inds][self.seq_start_indices].swapaxes(0, 1),
        )
        snn_states_pi = self.to_torch(snn_states_pi[0]).contiguous()
        snn_states_vf = self.to_torch(snn_states_vf[0]).contiguous()
        snn_extractor_pi = self.to_torch(snn_extractor_pi[0]).contiguous()
        snn_extractor_vf = self.to_torch(snn_extractor_vf[0]).contiguous()

        observations = {key: self.pad(obs[batch_inds]) for (key, obs) in self.observations.items()}
        observations = {key: obs.reshape((padded_batch_size,) + self.obs_shape[key]) for (key, obs) in observations.items()}

        return RecurrentDictRolloutBufferSamples(
            observations=observations,
            actions=self.pad(self.actions[batch_inds]).reshape((padded_batch_size,) + self.actions.shape[1:]),
            old_values=self.pad_and_flatten(self.values[batch_inds]),
            old_log_prob=self.pad_and_flatten(self.log_probs[batch_inds]),
            advantages=self.pad_and_flatten(self.advantages[batch_inds]),
            returns=self.pad_and_flatten(self.returns[batch_inds]),
            lstm_states=SNNStates(pi_potentials=snn_states_pi, vf_potentials=snn_states_vf, pi_extractor=snn_extractor_pi, vf_extractor=snn_extractor_vf),
            episode_starts=self.pad_and_flatten(self.episode_starts[batch_inds]),
            mask=self.pad_and_flatten(np.ones_like(self.returns[batch_inds])),
        )
