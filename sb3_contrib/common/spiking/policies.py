from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import zip_strict
from torch import nn

from sb3_contrib.common.spiking.type_aliases import SNNStates

from sb3_contrib.common.spiking.snn_nets import SNN
# from sb3_contrib.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, NatureCNN, CombinedExtractor

class SNNExtractor(nn.Module):
    """
    Constructs an MLP that receives the output from a previous features extractor (i.e. a CNN) or directly
    the observations (if no features extractor is applied) as an input and outputs a latent representation
    for the policy and a value network.

    The ``net_arch`` parameter allows to specify the amount and size of the hidden layers.
    It can be in either of the following forms:
    1. ``dict(vf=[<list of layer sizes>], pi=[<list of layer sizes>])``: to specify the amount and size of the layers in the
        policy and value nets individually. If it is missing any of the keys (pi or vf),
        zero layers will be considered for that key.
    2. ``[<list of layer sizes>]``: "shortcut" in case the amount and size of the layers
        in the policy and value nets are the same. Same as ``dict(vf=int_list, pi=int_list)``
        where int_list is the same for the actor and critic.

    .. note::
        If a key is not specified or an empty list is passed ``[]``, a linear network will be used.

    :param feature_dim: Dimension of the feature vector (can be the output of a CNN)
    :param net_arch: The specification of the policy and value networks.
        See above for details on its formatting.
    :param activation_fn: The activation function to use for the networks.
    :param device: PyTorch device.
    """

    def __init__(
        self,
        input_size: int,
        output_dim: int,
        device: Union[th.device, str] = "auto",
    ) -> None:
        super().__init__()
        policy_net: List[nn.Module] = []
        value_net: List[nn.Module] = []


        self.policy_net = SNN(input_size, output_dim, snn_hidden_size=None, n_hidden=0)

        self.value_net = SNN(input_size, 1, snn_hidden_size=None, n_hidden=0)

        # Save dim, used to create the distributions
        self.latent_dim_pi = output_dim
        self.latent_dim_vf = 1

    def init_mem(self):
        self.policy_net.init_mem()
        self.value_net.init_mem()

    def forward(self, features: th.Tensor, states:SNNStates) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features,states), self.forward_critic(features,states)

    def forward_actor(self, features: th.Tensor, states:SNNStates) -> th.Tensor:
        return self.policy_net(features,states.pi_extractor)

    def forward_critic(self, features: th.Tensor, states:SNNStates) -> th.Tensor:
        return self.value_net(features,states.vf_extractor)

class LinExtractor(nn.Module):
    """
    Constructs an MLP that receives the output from a previous features extractor (i.e. a CNN) or directly
    the observations (if no features extractor is applied) as an input and outputs a latent representation
    for the policy and a value network.

    The ``net_arch`` parameter allows to specify the amount and size of the hidden layers.
    It can be in either of the following forms:
    1. ``dict(vf=[<list of layer sizes>], pi=[<list of layer sizes>])``: to specify the amount and size of the layers in the
        policy and value nets individually. If it is missing any of the keys (pi or vf),
        zero layers will be considered for that key.
    2. ``[<list of layer sizes>]``: "shortcut" in case the amount and size of the layers
        in the policy and value nets are the same. Same as ``dict(vf=int_list, pi=int_list)``
        where int_list is the same for the actor and critic.

    .. note::
        If a key is not specified or an empty list is passed ``[]``, a linear network will be used.

    :param feature_dim: Dimension of the feature vector (can be the output of a CNN)
    :param net_arch: The specification of the policy and value networks.
        See above for details on its formatting.
    :param activation_fn: The activation function to use for the networks.
    :param device: PyTorch device.
    """

    def __init__(
        self,
        input_size: int,
        output_dim: int,
        device: Union[th.device, str] = "auto",
    ) -> None:
        super().__init__()
        policy_net: List[nn.Module] = []
        value_net: List[nn.Module] = []


        self.policy_net = nn.Linear(input_size, output_dim)

        self.value_net = nn.Linear(input_size, 1)

        # Save dim, used to create the distributions
        self.latent_dim_pi = output_dim
        self.latent_dim_vf = output_dim


    def forward(self, features: th.Tensor, states:SNNStates) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features,states), self.forward_critic(features,states)

    def forward_actor(self, features: th.Tensor, states:SNNStates) -> th.Tensor:
        return self.policy_net(features,states.pi_potentials)

    def forward_critic(self, features: th.Tensor, states:SNNStates) -> th.Tensor:
        return self.value_net(features,states.vf_potentials)


class SpikingActorCriticPolicy(ActorCriticPolicy):
    """
    Spiking NN policy class for actor-critic algorithms (has both policy and value prediction).
    To be used with A2C, PPO and the likes.
    It assumes that both the actor and the critic SNN
    have the same architecture.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param share_features_extractor: If True, the features extractor is shared between the policy and value networks.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param snn_hidden_size: Number of hidden neurons for each SNN layer.
    :param n_snn_layers: Number of SNN layers.
    :param shared_SNN: Whether the SNN is shared between the actor and the critic
        (in that case, only the actor gradient is used)
        By default, the actor and the critic have two separate SNN.
    :param enable_critic_SNN: Use a seperate SNN for the critic.
    :param SNN_kwargs: Additional keyword arguments to pass the the SNN
        constructor.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        snn_hidden_size: int = 32,
        n_snn_layers: int = 1,
        shared_snn: bool = True,
        enable_critic_snn: bool = False,
        snn_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.snn_output_dim = snn_hidden_size
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

        self.snn_kwargs = snn_kwargs or {}
        self.shared_snn = shared_snn
        self.enable_critic_snn = enable_critic_snn
        self.snn_actor = SNN(
            input_size=self.features_dim, # input size
            output_dim = snn_hidden_size, # output size
            n_hidden = 1,
            snn_hidden_size=snn_hidden_size,
            random_beta=True,
            **self.snn_kwargs,
        )
        # For the predict() method, to initialize hidden states
        # (n_snn_layers, batch_size, snn_hidden_size)
        print("snn hidden state shape is n_snn_layers, batch_size, snn_hidden_size. n_snn_layers is 2, batch_size is 1, snn_hidden_size is user input")
        self.snn_hidden_state_shape = (2, 1, snn_hidden_size)
        self.critic = None
        self.snn_critic = None
        assert not (
            self.shared_snn and self.enable_critic_snn
        ), "You must choose between shared SNN, seperate or no SNN for the critic."

        assert not (
            self.shared_snn and not self.share_features_extractor
        ), "If the features extractor is not shared, the SNN cannot be shared."

        # No SNN for the critic, we still need to convert
        # output of features extractor to the correct size
        # (size of the output of the actor snn)
        if not (self.shared_snn or self.enable_critic_snn):
            self.critic = nn.Linear(self.features_dim, snn_hidden_size)

        # Use a separate SNN for the critic
        if self.enable_critic_snn:
            self.snn_critic = SNN(
                self.features_dim, # input size
                snn_hidden_size, # output size
                num_layers=n_snn_layers,
                **self.snn_kwargs,
            )

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # print("Building MLP extractor")
        self.mlp_extractor = SNNExtractor(self.snn_output_dim,self.action_space.n)

    def init_mem(self):
        self.snn_actor.init_mem()
        if self.snn_critic is not None:
            self.snn_critic.init_mem()
        self.mlp_extractor.init_mem()

    @staticmethod
    def _process_sequence(
        features: th.Tensor,
        snn_states: th.Tensor,
        episode_starts: th.Tensor,
        snn: SNN,
        return_all: bool = True,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Do a forward pass in the SNN network.

        :param features: Input tensor
        :param snn_states: previous membrane potentials of the SNN
        :param episode_starts: Indicates when a new episode starts,
            in that case, we need to reset SNN states.
        :param snn: SNN object.
        :param return_all: Whether to return all the membrane potentials upon the forward pass. Required to pass to the SNN extractor
        :return: SNN output and updated SNN states.
        """
        # SNN logic
        # (sequence length, batch size, features dim)
        # (batch size = n_envs for data collection or n_seq when doing gradient update)
        n_seq = snn_states.shape[1]
        
        # Batch to sequence
        # (padded batch size, features_dim) -> (n_seq, max length, features_dim) -> (max length, n_seq, features_dim)
        # note: max length (max sequence length) is always 1 during data collection
        features_sequence = features.reshape((n_seq, -1, snn.input_size)).swapaxes(0, 1)
        episode_starts = episode_starts.reshape((n_seq, -1)).swapaxes(0, 1)

        # If we don't have to reset the state in the middle of a sequence
        # we can avoid the for loop, which speeds up things
        if th.all(episode_starts == 0.0):
            snn_output, snn_states = snn(features_sequence, snn_states,return_all=return_all)
            snn_output = th.flatten(snn_output.transpose(0, 1), start_dim=0, end_dim=1)
            return snn_output, snn_states

        snn_output = []

        # Iterate over the sequence
        for features, episode_start in zip_strict(features_sequence, episode_starts):
            hidden, snn_states = snn(
                features.unsqueeze(dim=0),
                
                    # Reset the states at the beginning of a new episode, if new episode, episode_start is 1
                    # if not new episode, snn_states is multiplied by 1.0 and is thus previous snn_states
                    (1.0 - episode_start).view(1, n_seq, 1) * snn_states,
                    return_all=return_all,
                
            )
            snn_output += [hidden]
        # Sequence to batch
        # (sequence length, n_seq, snn_out_dim) -> (batch_size, snn_out_dim)
        snn_output = th.flatten(th.cat(snn_output).transpose(0, 1), start_dim=0, end_dim=1)
        # snn_output = th.cat(snn_output).transpose(0, 1)
        # if return_all:
        #     snn_output = th.cat(snn_output,dim=0).reshape((-1,snn.output_dim))

        # else:
        #     snn_output = th.flatten(th.cat(snn_output).transpose(0, 1), start_dim=0, end_dim=1)
        return snn_output, snn_states

    def forward(
        self,
        obs: th.Tensor,
        snn_states: SNNStates,
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, SNNStates]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation. Observation (n_envs, obs_shape)
        :param snn_states: The last hidden states for the SNN.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the snn states in that case).
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features  # alis
        else:
            pi_features, vf_features = features
        # latent_pi, latent_vf = self.mlp_extractor(features)

        latent_pi, snn_states_pi = self._process_sequence(pi_features, snn_states.pi_potentials, episode_starts, self.snn_actor, return_all=True)
        if self.snn_critic is not None:
            latent_vf, snn_states_vf = self._process_sequence(vf_features, snn_states.vf_potentials, episode_starts, self.snn_critic,return_all=True)
        elif self.shared_snn:
            # Re-use SNN features but do not backpropagate
            latent_vf = latent_pi.detach()
            snn_states_vf = (snn_states_pi.detach()) # changed from snn_states_pi[0].detach()
        else:
            # Critic only has a feedforward network
            latent_vf = self.critic(vf_features)
            snn_states_vf = snn_states_pi

        # I THINK SOMETHING IS GOING WRONG HERE...
        # latent_pi, pi_extractor = self.mlp_extractor.forward_actor(latent_pi, snn_states)
        # values, vf_extractor = self.mlp_extractor.forward_critic(latent_vf, snn_states)

        latent_pi, pi_extractor = self._process_sequence(latent_pi, snn_states.pi_extractor, episode_starts, self.mlp_extractor.policy_net,return_all=True)
        values, vf_extractor = self._process_sequence(latent_vf, snn_states.vf_extractor, episode_starts, self.mlp_extractor.value_net,return_all=True)

        # Sequence to batch
        # (sequence length, n_seq, snn_out_dim) -> (batch_size, snn_out_dim) was with the original thing in the process_sequence function
        # latent_pi = th.flatten(latent_pi.transpose(0, 1), start_dim=0, end_dim=1)
        # values = th.flatten(values.transpose(0, 1), start_dim=0, end_dim=1)
        # Evaluate the values for the given observations
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        return actions, values, log_prob, SNNStates(pi_potentials=snn_states_pi,
                                                    vf_potentials= snn_states_vf,
                                                    pi_extractor=pi_extractor,
                                                    vf_extractor=vf_extractor)

    def get_distribution(
        self,
        obs: th.Tensor,
        snn_states: Tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> Tuple[Distribution, Tuple[th.Tensor, ...]]:
        """
        Get the current policy distribution given the observations.

        :param obs: Observation.
        :param snn_states: The last potential states for the SNN.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the snn states in that case).
        :return: the action distribution and new hidden states.
        """
        # Call the method from the parent of the parent class
        features = super(ActorCriticPolicy, self).extract_features(obs, self.pi_features_extractor)
        latent_pi, snn_states = self._process_sequence(features, snn_states, episode_starts, self.snn_actor)

        return self._get_action_dist_from_latent(latent_pi), snn_states

    def predict_values(
        self,
        obs: th.Tensor,
        snn_states: Tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
    ) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation.
        :param snn_states: The last potential states for the SNN. First vf_potentials of main SNN, second vf_extractor of mlp extractor SNN.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the snn states in that case).
        :return: the estimated values.
        """
        wrapped_states = SNNStates(pi_potentials=th.tensor(0),pi_extractor=th.tensor(0),vf_potentials=snn_states[0], vf_extractor=snn_states[1])
        # Call the method from the parent of the parent class
        features = super(ActorCriticPolicy, self).extract_features(obs, self.vf_features_extractor)

        if self.snn_critic is not None:
            latent_vf, snn_states_vf = self._process_sequence(features, wrapped_states.vf_potentials, episode_starts, self.snn_critic)
        elif self.shared_snn:
            # Use SNN from the actor
            latent_pi, _ = self._process_sequence(features, wrapped_states.vf_potentials, episode_starts, self.snn_actor)
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(features)


        # return self.value_net(latent_vf)
        
        # values, snn_states = self.mlp_extractor.forward_critic(latent_vf,wrapped_states)
        # values, snn_states = self._process_sequence(latent_vf,wrapped_states)
        values, vf_extractor = self._process_sequence(latent_vf, wrapped_states.vf_extractor, episode_starts, self.mlp_extractor.value_net,return_all=True)
        return values


    def evaluate_actions(
        self, obs: th.Tensor, actions: th.Tensor, snn_states: SNNStates, episode_starts: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation.
        :param actions:
        :param snn_states: The last potential states for the SNN.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the snn states in that case).
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features  # alias
        else:
            pi_features, vf_features = features
        latent_pi, _ = self._process_sequence(pi_features, snn_states.pi_potentials, episode_starts, self.snn_actor)
        if self.snn_critic is not None:
            latent_vf, _ = self._process_sequence(vf_features, snn_states.vf_potentials, episode_starts, self.snn_critic)
        elif self.shared_snn:
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(vf_features)


        # latent_pi, _= self.mlp_extractor.forward_actor(latent_pi,snn_states)
        # latent_vf, _ = self.mlp_extractor.forward_critic(latent_vf,snn_states)
        latent_pi, _ = self._process_sequence(latent_pi, snn_states.pi_extractor, episode_starts, self.mlp_extractor.policy_net,return_all=True)        
        latent_vf, vf_extractor = self._process_sequence(latent_vf, snn_states.vf_extractor, episode_starts, self.mlp_extractor.value_net,return_all=True)


        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def _predict(
        self,
        observation: th.Tensor,
        snn_states: Tuple[th.Tensor, th.Tensor],
        episode_starts: th.Tensor,
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, Tuple[th.Tensor, ...]]:
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param snn_states: The last potential states for the SNN.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the snn states in that case).
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy and hidden states of the SNN
        """
        distribution, snn_states = self.get_distribution(observation, snn_states, episode_starts)
        return distribution.get_actions(deterministic=deterministic), snn_states

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param snn_states: The last potential states for the SNN.
        :param episode_starts: Whether the observations correspond to new episodes
            or not (we reset the snn states in that case).
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        observation, vectorized_env = self.obs_to_tensor(observation)

        if isinstance(observation, dict):
            n_envs = observation[next(iter(observation.keys()))].shape[0]
        else:
            n_envs = observation.shape[0]
        # state : (n_layers, n_envs, dim)
        if state is None:
            # Initialize hidden states to zeros
            state = np.concatenate([np.zeros(self.snn_hidden_state_shape) for _ in range(n_envs)], axis=1)

        if episode_start is None:
            episode_start = np.array([False for _ in range(n_envs)])

        with th.no_grad():
            # Convert to PyTorch tensors
            states = th.tensor(state, dtype=th.float32, device=self.device)
            episode_starts = th.tensor(episode_start, dtype=th.float32, device=self.device)
            actions, states = self._predict(
                observation, snn_states=states, episode_starts=episode_starts, deterministic=deterministic
            )
            states = (states[0].cpu().numpy(), states[1].cpu().numpy())

        # Convert to numpy
        actions = actions.cpu().numpy()

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)

        return actions, states

