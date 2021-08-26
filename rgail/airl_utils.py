# ===========================================================================
# AIRL Ref1: https://github.com/uidilr/deepirl_chainer/blob/master/irl/airl/discriminator.py
# AIRL Ref2: https://github.com/ahq1993/inverse_rl/blob/master/inverse_rl/models/eairl.py
# ===========================================================================

import os
from itertools import accumulate
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import stable_baselines3.common.callbacks as callbacks
import torch as th
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.utils import update_learning_rate
from stable_baselines3.common.running_mean_std import RunningMeanStd
from torch import nn
from tqdm import tqdm


# ===========================================================================
# Attention module
# ===========================================================================
#class Attention(nn.Module):
#    def __init__(self, T, K_in, K_out):
#        """
#        Args:
#            T (int): Sequence Length
#            K_in (int): Vector (embedding) length of input vectors
#            K_out (int): Vector (embedding) length of output vectors
#        """
#        super(Attention, self).__init__()
#


# ===========================================================================
# Return Appropriate Discriminator Object
# ===========================================================================
def get_discriminator(algorithm: str,
                      base_disc_params,
                      gail_disc_params,
                      airl_disc_params):
    if algorithm == 'gail':
        base_disc_params.update(gail_disc_params)
        return GAILDiscriminator(**base_disc_params)
    elif algorithm == 'airl':
        base_disc_params.update(airl_disc_params)
        return AIRLDiscriminator(**base_disc_params)
    else:
        raise NotImplementedError

# ===========================================================================
# Base Discriminator Class
# ===========================================================================

class BaseDiscriminator(nn.Module):
    def __init__(
            self,
            obs_dim: int,
            acs_dim: int,
            is_discrete: bool,
            batch_size: int,
            lr_schedule: Callable[[float], float],
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            eps: float = 1e-5,
            device: str = "cpu"
        ):
        super(BaseDiscriminator, self).__init__()

        self.obs_dim = obs_dim
        self.acs_dim = acs_dim
        self.is_discrete = is_discrete
        self.input_dims = self.get_input_dims()

        self.batch_size = batch_size
        self.is_discrete = is_discrete
        self.device = device
        self.eps = eps

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer_class = optimizer_class
        self.lr_schedule = lr_schedule

        self.current_progress_remaining = 1.

        self._build()

    def _sub_class_vars_to_save(self):
        """
        Returns a list of variable names(str) that we want to save
        that are specific to sub-class instance.
        """
        raise NotImplementedError

    def get_input_dims(self):
        raise NotImplementedError

    def _build_networks(self):
        raise NotImplementedError

    def _get_networks_dict(self):
        # return a dictionary containing network
        # name and its state_dict
        raise NotImplementedError

    def _load_networks_from_dict(self, state_dict):
        # load the networks from the dict
        raise NotImplementedError

    def _loss_fn(self, preds, nominal):
        raise NotImplementedError

    def forward(self,
                current_states: th.tensor,
                next_states: th.tensor,
                actions: th.tensor,
                log_action_probs: th.tensor) -> th.tensor:
        raise NotImplementedError

    def reward_function(self,
                        current_states: th.tensor,
                        next_states: th.tensor,
                        actions: th.tensor) -> np.ndarray:
        raise NotImplementedError

    def _build_optimizer(self):
        raise NotImplementedError

    def _get_optimizers_dict(self):
        raise NotImplementedError

    def _load_optimizers_from_dict(self, state_dict):
        raise NotImplementedError

    def _build(self) -> None:
        # Create networks
        self._build_networks()

        # Build optimizer
        self._build_optimizer()

    def train(
            self,
            iterations: np.ndarray,
            nominal_data: Dict[str, th.tensor],
            expert_data: Dict[str, th.tensor],
            current_progress_remaining: float = 1,
        ) -> Dict[str, Any]:

        print(nominal_data.keys())
        print(expert_data.keys())
        # Update learning rate
        self._update_learning_rate(current_progress_remaining)

        for itr in tqdm(range(iterations)):
            for batch_indices in self.get(min(nominal_data['dataset_size'], expert_data['dataset_size'])):
                # Make predictions
                nominal_preds = self.forward(current_states   = nominal_data['current_states'][batch_indices, ...],
                                             next_states      = nominal_data['next_states'][batch_indices, ...],
                                             actions          = nominal_data['actions'][batch_indices, ...],
                                             log_action_probs = nominal_data['log_action_probs'][batch_indices, ...])
                expert_preds = self.forward(current_states    = expert_data['current_states'][batch_indices, ...],
                                            next_states       = expert_data['next_states'][batch_indices, ...],
                                            actions           = expert_data['actions'][batch_indices, ...],
                                            log_action_probs  = expert_data['log_action_probs'][batch_indices, ...])

                # Calculate loss
                nominal_loss = self._loss_fn(nominal_preds, nominal=True)
                expert_loss = self._loss_fn(expert_preds, nominal=False)
                loss = nominal_loss + expert_loss

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        disc_metrics =  {"discriminator/disc_loss": loss.item(),
                         "discriminator/expert_loss": expert_loss.item(),
                         "discriminator/nominal_loss": nominal_loss.item(),
                         "discriminator/mean_nominal_preds": nominal_preds.mean().item(),
                         "discriminator/max_nominal_preds": nominal_preds.max().item(),
                         "discriminator/min_nominal_preds": nominal_preds.min().item(),
                         "discriminator/mean_expert_preds": expert_preds.mean().item(),
                         "discriminator/max_expert_preds": expert_preds.max().item(),
                         "discriminator/min_expert_preds": expert_preds.min().item(),
                       }
        return disc_metrics

    def get(self, size: int) -> np.ndarray:
        indices = np.random.permutation(size)

        batch_size = self.batch_size
        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = size

        start_idx = 0
        while start_idx < size:
            batch_indices = indices[start_idx:start_idx+batch_size]
            yield batch_indices
            start_idx += batch_size

    def _update_learning_rate(self, current_progress_remaining) -> None:
        self.current_progress_remaining = current_progress_remaining
        update_learning_rate(self.optimizer, self.lr_schedule(current_progress_remaining))

    def _base_class_vars_to_save(self):
        base_variables_list = [
                'obs_dim',
                'acs_dim',
                'is_discrete',
                'obs_select_dim',
                'acs_select_dim',
                'device',
                'hidden_sizes',]
        return base_variables_list

    def _vars_to_save(self):
        base_class_vars = self._base_class_vars_to_save()
        sub_class_vars = self._sub_class_vars_to_save()
        if sub_class_vars is None:
            return base_class_vars
        else:
            return base_class_vars + sub_class_vars

    def save(self, save_path):
        vars_to_save, variables_dict = self._vars_to_save(), {}
        for k in vars_to_save:
                variables_dict[k] = getattr(self, k)
        networks_dict = self._get_networks_dict()
        optimizers_dict = self._get_optimizers_dict()
        state_dict = variables_dict + networks_dict + optimizers_dict
        # Attempt to save variables that are specific to sub-class
        th.save(state_dict, save_path)

    def _load(self, load_path, skip_variables):
        state_dict = th.load(load_path)
        # Load variables
        vars_to_load = self._vars_to_save()# We can only load vars that were saved
        for var in vars_to_load and var not in skip_variables:
            setattr(self, var, state_dict[var])
        # rebuild just in case
        self._build()
        # Load networks and optimizers
        self._load_networks_from_dict(state_dict)
        self._load_optimizers_from_dict(state_dict)


# ===========================================================================
# GAIL Discriminator
# ===========================================================================

class GAILDiscriminator(BaseDiscriminator):
    def __init__(
            self,
            obs_dim: int,
            acs_dim: int,
            hidden_sizes: Tuple[int, ...],
            batch_size: int,
            lr_schedule: Callable[[float], float],
            is_discrete: bool,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            eps: float = 1e-5,
            device: str = "cpu"
        ):
        self.hidden_sizes = hidden_sizes
        super(GAILDiscriminator, self).__init__(
            obs_dim=obs_dim,
            acs_dim=acs_dim,
            is_discrete=is_discrete,
            batch_size=batch_size,
            lr_schedule=lr_schedule,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            eps=eps,
            device=device)

    def _sub_class_vars_to_save(self):
        """
        Returns a list of variable names(str) that we want to save
        that are specific to sub-class instance.
        Or returns None.
        """
        sub_class_vars = ['hidden_sizes']
        return sub_class_vars

    def get_input_dims(self):
        return self.obs_dim + self.acs_dim

    def _build_networks(self):
        # Create network and add sigmoid at the end
        self.network = nn.Sequential(
                *create_mlp(self.input_dims, 1, self.hidden_sizes),
                nn.Sigmoid()
        )
        self.network.to(self.device)

    def _get_networks_dict(self):
        """Returns a dictionary containing network name and its state_dict"""
        return dict(network=self.network.state_dict())

    def _load_networks_from_dict(self, state_dict):
        self.network.load_state_dict(state_dict['network'])

    def _build_optimizer(self):
        if self.optimizer_class is not None:
            self.optimizer = self.optimizer_class(self.parameters(),
                                                  lr=self.lr_schedule(1),
                                                  **self.optimizer_kwargs)
        else:
            raise NotImplementedError

    def _get_optimizers_dict(self):
        return dict(optimizer=self.optimizer.state_dict())

    def _load_optimizers_from_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def _loss_fn(self, preds, nominal):
        if not hasattr(self, 'criterion'):
            self.criterion = nn.BCELoss()
        if nominal is True:
            loss = self.criterion(preds, th.zeros(*preds.size()))
        else:
            loss = self.criterion(preds, th.ones(*preds.size()))
        return loss

    def forward(self,
                current_states: th.tensor,
                next_states: th.tensor,
                actions: th.tensor,
                log_action_probs: th.tensor) -> th.tensor:
        preds = self.network(th.cat([current_states, actions], dim=-1))
        return preds

    def reward_function(self,
                        current_states: th.tensor,
                        next_states: th.tensor,
                        actions: th.tensor) -> np.ndarray:
        with th.no_grad():
            preds = self.network(th.cat([current_states, actions], dim=-1))
        return np.log(preds.numpy()+self.eps)


# ===========================================================================
# AIRL Discriminator
# ===========================================================================

class AIRLDiscriminator(BaseDiscriminator):
    def __init__(
            self,
            obs_dim: int,
            acs_dim: int,
            reward_net_hidden_sizes: Tuple[int, ...],
            value_net_hidden_sizes: Tuple[int, ...],
            batch_size: int,
            lr_schedule: Callable[[float], float],
            is_discrete: bool,
            gamma: float, # discount factor
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            eps: float = 1e-5,
            reward_scheme: str = 's',    # 's' = state only reward
                                         # 'ss' = current and next state only reward
                                         # 'sa' = current state and current action reward
            device: str = "cpu"
        ):
        self.reward_net_hidden_sizes = reward_net_hidden_sizes
        self.value_net_hidden_sizes = value_net_hidden_sizes
        self.gamma = gamma
        self.reward_scheme = reward_scheme

        super(AIRLDiscriminator, self).__init__(
            obs_dim=obs_dim,
            acs_dim=acs_dim,
            is_discrete=is_discrete,
            batch_size=batch_size,
            lr_schedule=lr_schedule,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            eps=eps,
            device=device
            )
    def _sub_class_vars_to_save(self):
        """
        Returns a list of variable names(str) that we want to save
        that are specific to sub-class instance.
        """
        class_vars = ['reward_net_hidden_sizes',
                      'value_net_hidden_sizes',
                      'reward_scheme',
                      'gamma',]
        return class_vars

    def get_input_dims(self):
        if self.reward_scheme == 's':
            input_dims = self.obs_dim
        elif self.reward_scheme == 'ss':
            input_dims = self.obs_dim + self.obs_dim
        elif self.reward_scheme == 'sa':
            input_dims = self.obs_dim + self.acs_dim
        return input_dims

    def _build_networks(self):
        # Create networks
        self.reward_net = nn.Sequential(
                *create_mlp(self.input_dims, 1, self.reward_net_hidden_sizes),
        )
        self.value_net = nn.Sequential(
                *create_mlp(self.obs_dim, 1, self.reward_net_hidden_sizes),
        )
        # Transfer to device
        self.reward_net.to(self.device)
        self.value_net.to(self.device)

    def _get_networks_dict(self):
        # return a dictionary containing network
        # name and its state_dict
        return dict(reward_net=self.reward_net.state_dict(),
                    value_net=self.value_net.state_dict())

    def _load_networks_from_dict(self, state_dict):
        # load the networks from the dict
        self.reward_net.load_state_dict(state_dict['reward_net'])
        self.value_net.load_state_dict(state_dict['value_net'])

    def _build_optimizer(self):
        # Build optimizer
        if self.optimizer_class is not None:
        #    self.reward_optim = self.optimizer_class(self.reward_net.parameters())
        #    self.value_optim = self.optimizer_class(self.reward_net.parameters())
            self.optimizer = self.optimizer_class(self.parameters(),
                                                  lr=self.lr_schedule(1),
                                                  **self.optimizer_kwargs)
        else: # TODO: should not this be a warning or error?
            self.optimizer = None

    def _get_optimizers_dict(self):
        return dict(optimizer=self.optimizer.state_dict())

    def _load_optimizers_from_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def _loss_fn(self, preds, nominal):
        if not hasattr(self, 'criterion'):
            self.criterion = nn.BCELoss()
        if nominal is True:
            loss = self.criterion(preds, th.zeros(*preds.size()))
        else:
            loss = self.criterion(preds, th.ones(*preds.size()))
        return loss

    def forward(self,
                current_states: th.tensor,
                next_states: th.tensor,
                actions: th.tensor,
                log_action_probs: th.tensor) -> th.tensor:
        if self.reward_scheme == 's':
            rewards = self.reward_net(current_states)
        elif self.reward_scheme == 'ss':
            rewards = self.reward_net(th.cat([current_states, next_states], dim = -1))
        elif self.reward_scheme == 'sa':
            rewards = self.reward_net(th.cat([current_states, actions], dim=-1))
        values = self.value_net(current_states)
        next_values = self.value_net(next_states)
        shaped_reward = rewards + self.gamma*next_values - values

        return (shaped_reward - th.logsumexp(th.cat([shaped_reward,
                                                          log_action_probs],
                                                          dim=1),
                                               dim=1, keepdim=True))

    def reward_function(self,
                        current_states: th.tensor,
                        next_states: th.tensor,
                        actions: th.tensor) -> np.ndarray:
        with th.no_grad():
            if self.reward_scheme == 's':
                rewards = self.reward_net(current_states)
            elif self.reward_scheme == 'ss':
                rewards = self.reward_net(th.cat([current_states, next_states], dim = -1))
            elif self.reward_scheme == 'sa':
                rewards = self.reward_net(th.cat([current_states, actions], dim=-1))
        return rewards.numpy()

    def train(
            self,
            iterations: np.ndarray,
            nominal_data: Dict[str, th.tensor],
            expert_data: Dict[str, th.tensor],
            current_progress_remaining: float = 1,
        ) -> Dict[str, Any]:

        # Update learning rate
        self._update_learning_rate(current_progress_remaining)

        for itr in tqdm(range(iterations)):
            for batch_indices in self.get(min(nominal_data['dataset_size'], expert_data['dataset_size'])):
                # Make predictions
                nominal_preds = self.forward(nominal_data['current_states'][batch_indices, ...],
                                             nominal_data['next_states'][batch_indices, ...],
                                             nominal_data['actions'][batch_indices, ...],
                                             nominal_data['log_action_probs'][batch_indices, ...])
                expert_preds = self.forward(expert_data['current_states'][batch_indices, ...],
                                            expert_data['next_states'][batch_indices, ...],
                                            expert_data['actions'][batch_indices, ...],
                                            expert_data['log_action_probs'][batch_indices, ...])

                # Calculate loss
                #nominal_loss = self._loss_fn(nominal_preds, nominal=True)
                #expert_loss = self._loss_fn(expert_preds, nominal=False)
                loss = (-(nominal_preds + expert_preds)).mean()

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        disc_metrics =  {"discriminator/disc_loss": loss.item(),
                         #"discriminator/expert_loss": expert_loss.item(),
                         #"discriminator/nominal_loss": nominal_loss.item(),
                         "discriminator/mean_nominal_preds": nominal_preds.mean().item(),
                         "discriminator/max_nominal_preds": nominal_preds.max().item(),
                         "discriminator/min_nominal_preds": nominal_preds.min().item(),
                         "discriminator/mean_expert_preds": expert_preds.mean().item(),
                         "discriminator/max_expert_preds": expert_preds.max().item(),
                         "discriminator/min_expert_preds": expert_preds.min().item(),
                       }
        return disc_metrics


# =====================================================================================
# Inverse Dynamics Model
# =====================================================================================

class InverseDynamicsModel(nn.Module):
    """
    """
    def __init__(
            self,
            obs_dim: int,
            acs_dim: int,
            num_spurious_features: int,
            idm_latent_dim: int,
            discrete_actions: bool,
            idm_encoder_layers=[30,30],
            idm_inverse_model_layers=[60],
            idm_forward_model_layers=[60],
            idm_learning_rate=1e-3,
            idm_batch_size=64,
            idm_loss_type: str = 'fi',
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            agent: str = 'nominal', # nominal or expert
            device='cpu',
            ):
        super(InverseDynamicsModel, self).__init__()
        self.obs_dim = obs_dim + num_spurious_features
        self.acs_dim = acs_dim
        self.latent_dim = idm_latent_dim
        self.discrete_actions = discrete_actions
        if self.discrete_actions is True:
            raise NotImplementedError
        self.encoder_layers = idm_encoder_layers
        self.inverse_model_layers = idm_inverse_model_layers
        self.forward_model_layers = idm_forward_model_layers
        self.batch_size = idm_batch_size
        self.loss_type = idm_loss_type
        self.device = device

        self.agent = agent

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5
            optimizer_kwargs['lr'] = idm_learning_rate
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer_class = optimizer_class

        self._build()

    def _build(self):
        self.encoder = nn.Sequential(*create_mlp(self.obs_dim,
                                                 self.latent_dim,
                                                 self.encoder_layers))
        self.inverse_model = nn.Sequential(*create_mlp(self.latent_dim*2,
                                                   self.acs_dim,
                                                   self.inverse_model_layers))
        self.forward_model = nn.Sequential(*create_mlp(self.latent_dim + self.acs_dim,
                                                       self.latent_dim,
                                                       self.forward_model_layers))
        self.encoder.to(self.device)
        self.inverse_model.to(self.device)
        self.forward_model.to(self.device)

        self.loss_fn = nn.MSELoss()

        if self.optimizer_class is not None:
            self.optimizer = self.optimizer_class(self.parameters(), **self.optimizer_kwargs)
        else:
            self.optimizer = None

    def _init_conv_net(self, layers, last_layer_activation):
        raise NotImplementedError

    def encode(self, obs):
        return self.encoder(obs)

    def predict_action(self, obs_current, obs_next):
        x = th.cat([self.encode(obs_current),
                          self.encode(obs_next)],
                          axis=-1)
        return self.predictor(x)

    def update(self, data_dict, other_encoder=None):
        losses = []
        forward_losses = []
        inverse_losses = []
        grad_obs_current = 0
        grad_obs_next = 0
        i = 0
        for indices in self.get(data_dict['dataset_size']):
            obs_current = data_dict['current_states'][indices, ...]
            obs_next = data_dict['next_states'][indices, ...]
            true_acs = data_dict['actions'][indices, ...]

            # set the requires_grad True for the input to log its gradient
            obs_current.requires_grad = True
            obs_next.requires_grad = True

            encoded_oc = self.encode(obs_current)
            encoded_on = self.encode(obs_next)
            predicted_acs = self.inverse_model(th.cat([encoded_oc, encoded_on], axis=-1))
            predicted_on = self.forward_model(th.cat([encoded_oc, true_acs],axis=-1))

            loss_forward = self.loss_fn(predicted_on, encoded_on)
            loss_inverse = self.loss_fn(predicted_acs, true_acs)

            if 'f' in self.loss_type and 'i' in self.loss_type:
                loss = loss_forward + loss_inverse
            elif 'f' in self.loss_type:
                loss = loss_forward
            elif 'i' in self.loss_type:
                loss = loss_inverse
            else:
                raise NotImplementedError

            weight_tying = True
            wt_loss = th.tensor(0).float().to(self.device)
            if weight_tying and other_encoder is not None:
                for p_model, p_target in zip(self.encoder.parameters(),
                                             other_encoder.parameters()):
                    p_target = p_target.clone().detach()
                    wt_loss += th.sqrt(th.sum((p_model - p_target)**2))
                loss += wt_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            forward_losses.append(loss_forward.item())
            inverse_losses.append(loss_inverse.item())

            # log the gradients
            grad_obs_current += th.mean(th.abs(obs_current.grad.data), dim=0).numpy()
            grad_obs_next += th.mean(th.abs(obs_next.grad.data), dim=0).numpy()
            i += 1

        grad_obs_current /= i
        grad_obs_next /= i

        return {(self.agent + '_idm/idm_loss'): float(np.mean(losses)),
                (self.agent + '_idm/idm_forward_loss'): float(np.mean(forward_losses)),
                (self.agent + '_idm/idm_inverse_loss'): float(np.mean(inverse_losses)),
                (self.agent + '_idm/idm_wt_loss'): wt_loss.item(),
                (self.agent + '_idm/latent_dim'): self.latent_dim,
                (self.agent + '_idm/ratio_grad_current'): np.sum(grad_obs_current[:18])/np.sum(grad_obs_current[18:]),
                (self.agent + '_idm/ratio_grad_next'): np.sum(grad_obs_next[:18])/np.sum(grad_obs_next[18:]),
                (self.agent + '_idm/grad_current_real18'): np.sum(grad_obs_current[:18]),
                (self.agent + '_idm/grad_current_spurious'): np.sum(grad_obs_current[18:]),
                (self.agent + '_idm/grad_next_real18'): np.sum(grad_obs_next[:18]),
                (self.agent + '_idm/grad_next_spurious'): np.sum(grad_obs_next[18:]),}

    def get(self, size: int) -> np.ndarray:
        indices = np.random.permutation(size)
        batch_size = self.batch_size
        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = size
        start_idx = 0
        while start_idx < size:
            batch_indices = indices[start_idx:start_idx+batch_size]
            yield batch_indices
            start_idx += batch_size


# =====================================================================================
# AIRL CALLBACK
# =====================================================================================

class AIRLCallback(callbacks.BaseCallback):
    """Implements Adversarial Inverse Reinforcement Learning Algorithm."""
    def __init__(
        self,
        discriminator,
        expert_data,
        save_dir: str,
        plot_discriminator: bool,
        update_freq: int = 1,
        normalize_reward: bool = False,
        num_spurious_features: Optional[float] = None,
        use_inverse_dynamics_model: bool = False,
        inverse_dynamics_model_kwargs: Optional[Dict[str, Any]] = None,
        device: str = 'cpu',
        verbose: int = 1
    ):
        super(AIRLCallback, self).__init__(verbose)
        self.discriminator = discriminator
        self.expert_data = expert_data
        self.update_freq = update_freq
        self.plot_save_dir = save_dir
        self.plot_disc = plot_discriminator
        self.device = device

        # Spurious features
        self.num_spurious_features = num_spurious_features

        # Inverse Dynamics Model
        self.use_inverse_dynamics_model = use_inverse_dynamics_model
        self.inverse_dynamics_model_kwargs = inverse_dynamics_model_kwargs

        # Build inverse dynamics model if applicable
        if self.use_inverse_dynamics_model:
            self.inverse_dynamics_model_kwargs['agent'] = 'nominal'
            self.nominal_idm = InverseDynamicsModel(**self.inverse_dynamics_model_kwargs)
            self.inverse_dynamics_model_kwargs['agent'] = 'expert'
            self.expert_idm = InverseDynamicsModel(**self.inverse_dynamics_model_kwargs)

        # Steal some arguments from idm_kwargs
        self.is_discrete = self.inverse_dynamics_model_kwargs['discrete_actions']
        self.obs_dim = self.inverse_dynamics_model_kwargs['obs_dim']
        self.acs_dim = self.inverse_dynamics_model_kwargs['acs_dim']

        # Setup reward normalization
        self.norm_reward = normalize_reward
        self.ret_rms = RunningMeanStd(shape=())
        self.ret = None

    def _init_callback(self):
        self.disc_itr = 0
        from rgail.utils import del_and_make
        del_and_make(os.path.join(self.plot_save_dir, "Discriminator"))
        self.plot_folder = os.path.join(self.plot_save_dir, "Discriminator")
        #self.discriminator.plot_expert(os.path.join(self.plot_folder,'expert.png'))

    def _on_step(self):
        pass

    def _on_rollout_end(self):
        nominal_data = self._get_nominal_data()
        expert_data = self._get_expert_data()

        if self.use_inverse_dynamics_model:
            #nominal_and_expert_data = self._get_data_for_idm_update()
            nominal_idm_data, expert_idm_data = self._get_data_for_idm_update()
            expert_idm_metrics = self.expert_idm.update(expert_idm_data)
            nominal_idm_metrics = self.nominal_idm.update(nominal_idm_data,
                                                          self.expert_idm.encoder)
            for k, v in nominal_idm_metrics.items():
                self.logger.record(k, v)
            for k, v in expert_idm_metrics.items():
                k = k + '_e'
                self.logger.record(k, v)

            # Evaluat latent space alignment
            alignment_metrics = self.check_alignment()
            for k, v in alignment_metrics.items():
                self.logger.record(k, v)

        # update discriminator
        if self.disc_itr % self.update_freq == 0:
            metrics = self.discriminator.train(iterations=1,
                                               nominal_data=nominal_data,
                                               expert_data=expert_data)
            for k, v in metrics.items():
                self.logger.record(k,v)
        self._try_to_plot()

        # get rewards
        rewards_shape = self.model.rollout_buffer.rewards.shape
        disc_rewards = self.discriminator.reward_function(current_states = nominal_data['current_states'],
                                                          next_states    = nominal_data['next_states'],
                                                          actions        = nominal_data['actions'])
        rewards = self._add_envs_dim(disc_rewards,
                                     rewards_shape[0], rewards_shape[1]).squeeze()
        if self.norm_reward:
            normed_rewards = self._normalize_rewards(rewards)
            self.model.rollout_buffer.rewards = normed_rewards
            self.logger.record('discriminator/disc_rewards', np.mean(rewards)*1000)
            self.logger.record('discriminator/disc_normed_rewards', np.mean(normed_rewards)*1000)
        else:
            self.model.rollout_buffer.rewards = rewards
            self.logger.record('discriminator/disc_rewards', np.mean(rewards)*1000)

        # recompute returns and advantages
        last_values, dones = (self.model.extras['last_values'],
                              self.model.extras['dones'])
        self.model.rollout_buffer.compute_returns_and_advantage(last_values, dones)
        self.disc_itr +=1

    def _try_to_plot(self):
        #if self.plot_disc:
        #    save_name = os.path.join(self.plot_folder, str(self.disc_itr)+'.png')
        #    self.discriminator.plot(save_name, nominal_obs = unnormalized_obs)
        pass

    def _get_nominal_data(self, filter_obs=True):
        # Get data from buffer
        nominal_data = self._read_data_from_buffer()
        nominal_data = self._corrupt_data(nominal_data, 'nominal')
        nominal_data = self._prepare_data(nominal_data)
        nominal_data = self._filter_nominal_data(nominal_data)
        return nominal_data

    def _get_expert_data(self, filter_obs=True):
        expert_data = self.expert_data.copy()
        expert_data['log_action_probs'] = self.get_log_probs(expert_data['current_states'])
        expert_data = self._corrupt_data(expert_data, 'expert')
        expert_data = self._prepare_data(expert_data)
        expert_data = self._filter_expert_data(expert_data)
        return expert_data

    def _get_data_for_idm_update(self):
        nominal_data = self._read_data_from_buffer()
        expert_data = self.expert_data.copy()
        expert_data['log_action_probs'] = self.get_log_probs(expert_data['current_states'])

        # corrupt data
        nominal_data = self._corrupt_data(nominal_data, 'nominal')
        expert_data = self._corrupt_data(expert_data, 'expert')

        # prepare and concat data
        nominal_data = self._prepare_data(nominal_data)
        expert_data = self._prepare_data(expert_data)

#        nominal_and_expert_data = {}
#        for k in nominal_data.keys():
#            if k == 'dataset_size':
#                nominal_and_expert_data[k] = (nominal_data[k] + expert_data[k])
#            else:
#                nominal_and_expert_data[k] = th.cat([nominal_data[k],
#                                                     expert_data[k]],
#                                                    dim=0)
#        return nominal_and_expert_data
        return nominal_data, expert_data

    # Utility functions for processing data and getting it in appropriate shape and for
    # for the discriminator
    def get_log_probs(self, observations):
        with th.no_grad():
            obs_t = th.from_numpy(observations)
            actions, values, log_probs = self.model.policy.forward(obs_t)
            log_probs = log_probs.numpy()[:,None]
        return log_probs

    def _drop_envs_dim(self, x):
        """
        For arbitrary inputs of shape [buffer_size, n_envs, ...],
        it drops n_envs dimension and returns [buffer_size*n_envs, -1].
        """
        return np.reshape(x, (x.shape[0]*x.shape[1], -1))

    def _add_envs_dim(self, x, buffer_size, n_envs):
        """
        For arbitrary inputs of shape [batch_size*n_envs, ...],
        it [batch_size, n_envs, -1].
        """
        return np.reshape(x, (buffer_size, n_envs, -1))

    def _prepare_states(self, x):
        assert len(x.shape) == 2 and (x.shape[-1] == (self.obs_dim + self.num_spurious_features))
        # Normalize or clip obs here
        return th.from_numpy(x)

    def _prepare_actions(self, x):
        if self.is_discrete: # Handle discrete actions
            # Either there should be no action dimension or it should be one
            assert len(x.shape) == 1 or x.shape[-1] == 1
            x_ = x.astype(int)
            if len(x.shape) > 1:
                x_ = np.squeeze(x_, axis=-1)
            x = np.zeros([x.shape[0], self.acs_dim])
            x[np.arange(x_.shape[0]), x_] = 1.
            x = th.from_numpy(x)
        else:               # Handle continuous actions
            assert len(x.shape) == 2 and x.shape[-1] == self.acs_dim
            x = th.from_numpy(x)
        return x

    def _read_data_from_buffer(self):
        # Get data from buffer
        actions = self.model.rollout_buffer.actions.copy()
        log_probs = self.model.rollout_buffer.log_probs.copy()
        current_states = self.model.rollout_buffer.observations.copy()
        next_states = self.model.rollout_buffer.new_observations.copy()
        current_states = self.training_env.unnormalize_obs(current_states)
        next_states = self.training_env.unnormalize_obs(next_states)

        # Arrange nominal data
        nominal_data = {
                        'current_states': self._drop_envs_dim(current_states),
                        'actions': self._drop_envs_dim(actions),
                        'log_action_probs': self._drop_envs_dim(log_probs),
                        'next_states': self._drop_envs_dim(next_states),
                        }
        return nominal_data

    def _corrupt_data(
            self,
            data,
            expert_or_nominal):
        if self.num_spurious_features > 0:
            data['current_states'] = self.add_spurious_features(data['current_states'], expert_or_nominal,
                                                                self.num_spurious_features)
            data['next_states'] = self.add_spurious_features(data['next_states'], expert_or_nominal,
                                                                self.num_spurious_features)
        return data

    def _prepare_data(self, data):
        dataset_size = data['current_states'].shape[0]
        new_data = dict(current_states=self._prepare_states(data['current_states']).to(self.device).float(),
                        actions=self._prepare_actions(data['actions']).to(self.device).float(),
                        log_action_probs=th.from_numpy(data['log_action_probs']).to(self.device).float(),
                        next_states=self._prepare_states(data['next_states']).to(self.device).float(),
                        dataset_size=dataset_size)
        assert (new_data['current_states'].shape[0] == dataset_size and
                new_data['actions'].shape[0] == dataset_size and
                new_data['log_action_probs'].shape[0] == dataset_size and
                new_data['next_states'].shape[0] == dataset_size)
        return new_data

    def _filter_nominal_data(self, data):
        if self.use_inverse_dynamics_model:
            data['current_states'] = self.nominal_idm.encode(data['current_states']).detach()
            data['next_states'] = self.nominal_idm.encode(data['next_states']).detach()
        return data

    def _filter_expert_data(self, data):
        if self.use_inverse_dynamics_model:
            data['current_states'] = self.expert_idm.encode(data['current_states']).detach()
            data['next_states'] = self.expert_idm.encode(data['next_states']).detach()
        return data

    def add_spurious_features(
            self,
            x,
            y,  # string indicating nominal or expert
            num_features,   # number of spurious bits to add
            ):
        new_shape = list(x.shape)
        new_shape[-1] += num_features
        if y == 'expert':
            z = np.zeros(new_shape)
            z[...,:-num_features] = x
        elif y == 'nominal':
            z = np.ones(new_shape)
            z[...,:-num_features] = x
        return z

    def _normalize_rewards(self, rewards):
        buffer_size, n_envs = rewards.shape
        normalized_reward = np.zeros((buffer_size, n_envs))
        ret = np.zeros((n_envs,))
        dones = self.model.rollout_buffer.dones
        for i in range(buffer_size):
            r = rewards[i,:]
            ret = ret*self.training_env.gamma + r
            self.ret_rms.update(ret)
            # TODO: Should we clip reward too?
            normalized_reward[i,:] = r/np.sqrt(self.ret_rms.var + 1e-4)
        return normalized_reward

    def check_alignment(self):
        """
        Checks to what degree are the latent spaces of expert and nominal
        encoders aligned.
        """
        # Get some examples from buffer
        states = self._drop_envs_dim(self.model.rollout_buffer.observations.copy())
        # Prepare them for inputs to encoders
        states_nom = self.add_spurious_features(states, 'nominal', self.num_spurious_features)
        states_exp = self.add_spurious_features(states, 'expert', self.num_spurious_features)

        states_nom = self._prepare_states(states_nom).float().to(self.device)
        states_exp = self._prepare_states(states_exp).float().to(self.device)

        # Get outputs
        with th.no_grad():
            encoded_states_nom = self.nominal_idm.encode(states_nom)
            encoded_states_exp = self.expert_idm.encode(states_exp)
            # Get distance stats
            cosine_sim = nn.CosineSimilarity()(encoded_states_nom, encoded_states_exp).numpy()
            l2_dist = th.dist(encoded_states_nom,
                                 encoded_states_exp,
                                 2).numpy()
            l1_dist = th.dist(encoded_states_nom,
                                 encoded_states_exp,
                                 1).numpy()
        return {
                    'latent_dim/avg_cosine_sim':  np.mean(cosine_sim),
                    'latent_dim/max_cosine_sim':  np.max(cosine_sim),
                    'latent_dim/min_cosine_sim':  np.min(cosine_sim),
                    'latent_dim/avg_l2_dist':     np.mean(l2_dist),
                    'latent_dim/max_l2_dist':     np.max(l2_dist),
                    'latent_dim/min_l2_dist':     np.min(l2_dist),
                    'latent_dim/avg_l1_dist':     np.mean(l1_dist),
                    'latent_dim/max_l1_dist':     np.max(l1_dist),
                    'latent_dim/min_l1_dist':     np.min(l1_dist),
               }

