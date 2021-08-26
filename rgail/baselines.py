import os
from itertools import accumulate
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

from stable_baselines3.common.torch_layers import create_mlp
from rgail.airl_utils import BaseDiscriminator
import torch as th
import torch.nn as nn
import numpy as np

from tqdm import tqdm

class ThirdPersonDiscriminator(BaseDiscriminator):
    def __init__(
            self,
            obs_dim: int,
            acs_dim: int,
            latent_dim: int,
            feature_extractor_layers: Tuple[int, ...],
            reward_classifier_layers: Tuple[int, ...],
            domain_classifier_layers: Tuple[int, ...],
            batch_size: int,
            lr_schedule: Callable[[float], float],
            is_discrete: bool,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            eps: float = 1e-5,
            device: str = "cpu"
        ):
        self.latent_dim = latent_dim
        self.feature_extractor_layers = feature_extractor_layers
        self.reward_classifier_layers = reward_classifier_layers
        self.domain_classifier_layers = domain_classifier_layers
        super(ThirdPersonDiscriminator, self).__init__(
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
        sub_class_vars = ['latent_dim',
                          'feature_extractor_layers',
                          'reward_classifier_layers',
                          'domain_classifier_layers']
        return sub_class_vars

    def get_input_dims(self):
        return self.obs_dim + self.acs_dim

    def _build_networks(self):
        # Create network and add sigmoid at the end
        self.feature_extractor = nn.Sequential(
                *create_mlp(self.obs_dim, self.latent_dim, self.feature_extractor_layers),
        )
        self.reward_classifier = nn.Sequential(
                *create_mlp(self.latent_dim + self.acs_dim, 1, self.reward_classifier_layers),
                nn.Sigmoid())
        self.domain_classifier = nn.Sequential(
                *create_mlp(self.latent_dim + self.acs_dim, 1, self.domain_classifier_layers),
                nn.Sigmoid())
        self.feature_extractor.to(self.device)
        self.reward_classifier.to(self.device)
        self.domain_classifier.to(self.device)

    def _get_networks_dict(self):
        """Returns a dictionary containing network name and its state_dict"""
        return dict(feature_extractor=self.feature_extractor.state_dict(),
                    reward_classifier=self.reward_classifier.state_dict(),
                    domain_classifier=self.domain_classifier.state_dict(),)

    def _load_networks_from_dict(self, state_dict):
        self.feature_extractor.load_state_dict(state_dict['feature_extractor'])
        self.reward_classifier.load_state_dict(state_dict['reward_classifier'])
        self.domain_classifier.load_state_dict(state_dict['domain_classifier'])

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
        features = self.feature_extractor(current_states)
        reward_preds = self.reward_classifier(th.cat([features, actions], dim=-1))
        domain_preds = self.domain_classifier(th.cat([features, actions], dim=-1))
        return reward_preds, domain_preds

    def reward_function(self,
                        current_states: th.tensor,
                        next_states: th.tensor,
                        actions: th.tensor) -> np.ndarray:
        with th.no_grad():
            features = self.feature_extractor(current_states)
            preds = self.reward_classifier(th.cat([features, actions], dim=-1))
        return np.log(preds.numpy()+self.eps)

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
                nominal_r_preds, nominal_d_preds = self.forward(current_states   = nominal_data['current_states'][batch_indices, ...],
                                                                next_states      = nominal_data['next_states'][batch_indices, ...],
                                                                actions          = nominal_data['actions'][batch_indices, ...],
                                                                log_action_probs = nominal_data['log_action_probs'][batch_indices, ...])
                expert_r_preds, expert_d_preds = self.forward(current_states    = expert_data['current_states'][batch_indices, ...],
                                                              next_states       = expert_data['next_states'][batch_indices, ...],
                                                              actions           = expert_data['actions'][batch_indices, ...],
                                                              log_action_probs  = expert_data['log_action_probs'][batch_indices, ...])

                # Reward Classifier Loss
                reward_loss = self._loss_fn(nominal_r_preds, nominal=True) + self._loss_fn(expert_r_preds, nominal=False)
                domain_loss = self._loss_fn(nominal_d_preds, nominal=True) + self._loss_fn(expert_d_preds, nominal=False)
                loss = reward_loss - domain_loss

                # Update
                self.optimizer.zero_grad()
                loss.backward()

                # TODO: Correct the sign of domain classifier params
                for params in self.domain_classifier.parameters():
                    params.grad *= -1

                self.optimizer.step()

        disc_metrics =  {"discriminator/disc_loss": loss.item(),
                         "discriminator/reward_loss": reward_loss.item(),
                         "discriminator/domain_loss": domain_loss.item(),
                         "discriminator/mean_nominal_preds": nominal_r_preds.mean().item(),
                         "discriminator/max_nominal_preds": nominal_r_preds.max().item(),
                         "discriminator/min_nominal_preds": nominal_r_preds.min().item(),
                         "discriminator/mean_expert_preds": expert_r_preds.mean().item(),
                         "discriminator/max_expert_preds": expert_r_preds.max().item(),
                         "discriminator/min_expert_preds": expert_r_preds.min().item(),
                       }
        return disc_metrics


# ====================================================================================
# Variational Discriminator Bottleneck
# ====================================================================================

class VAILDiscriminator(BaseDiscriminator):
    def __init__(
            self,
            obs_dim: int,
            acs_dim: int,
            encoder_layers: Tuple[int, ...],
            reward_classifier_layers: Tuple[int, ...],
            vail_target_kl: int,
            vail_init_beta: float,
            batch_size: int,
            lr_schedule: Callable[[float], float],
            is_discrete: bool,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            eps: float = 1e-5,
            device: str = "cpu"
        ):
        self.encoder_layers = encoder_layers
        self.reward_classifier = reward_classifier_layers
        super(VAILDiscriminator, self).__init__(
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
        sub_class_vars = ['feature_extractor_layers',
                          'reward_classifier_layers',
                          'domain_classifier_layers']
        return sub_class_vars

    def get_input_dims(self):
        return self.obs_dim + self.acs_dim

    def _build_networks(self):
        # Create network and add sigmoid at the end
        self.feature_extractor = nn.Sequential(
                *create_mlp(self.input_dims, self.feature_extractor_layers, self.hidden_sizes),
        )
        self.reward_classifier = nn.Sequential(
                *create_mlp(self.latent_dim, 1, self.reward_classifier_layers),
                nn.Sigmoid())
        self.domain_classifier = nn.Sequential(
                *create_mlp(self.latent_dim, 1, self.domain_classifier_layers),
                nn.Sigmoid())
        self.feature_extractor.to(self.device)
        self.reward_classifier.to(self.device)
        self.domain_classifier.to(self.device)

    def _get_networks_dict(self):
        """Returns a dictionary containing network name and its state_dict"""
        return dict(feature_extractor=self.feature_extractor.state_dict(),
                    reward_classifier=self.reward_classifier.state_dict(),
                    domain_classifier=self.domain_classifier.state_dict(),)

    def _load_networks_from_dict(self, state_dict):
        self.feature_extractor.load_state_dict(state_dict['feature_extractor'])
        self.reward_classifier.load_state_dict(state_dict['reward_classifier'])
        self.domain_classifier.load_state_dict(state_dict['domain_classifier'])

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

    def kl_loss(self, mean, logstd):
        # https://github.com/qxcv/vdb-irl/blob/master/inverse_rl/utils/general.py#L78
        std = th.exp(logstd)
        kl_loss = 0.5*th.sum(-1 - 2*logstd + std**2 + mean**2, dim=-1)
        return kl_loss.mean()

    def beta_update(self, current_kl, target_kl):
        new_beta = self.beta + self.beta_step_size * (current_kl - target_kl)
        new_beta = th.maximum(0.0, new_beta)
        return new_beta.item()

    def _loss_fn(self, preds, nominal):
        if not hasattr(self, 'criterion'):
            self.criterion = nn.BCELoss()
        if nominal is True:
            loss = self.criterion(preds, th.zeros(*preds.size()))
        else:
            loss = self.criterion(preds, th.ones(*preds.size()))
        return loss

    def encode(self, x, sample=True):
        mean = self.encoder_mean(x)
        logstd = self.encoder_logstd(x)
        std = torch.exp(logstd)
        if sample:
            noise = th.random_normal(x.size)
            reparam = std * noise + mean
            return reparam, mean, logstd
        else:
            return mean, logstd

    def forward(self,
                current_states: th.tensor,
                next_states: th.tensor,
                actions: th.tensor,
                log_action_probs: th.tensor) -> th.tensor:
        features, mean, logstd = self.encode(current_states)
        reward_preds = self.reward_classifier(th.cat([features, actions], dim=-1))
        return reward_preds, mean, logstd

    def reward_function(self,
                        current_states: th.tensor,
                        next_states: th.tensor,
                        actions: th.tensor) -> np.ndarray:
        with th.no_grad():
            features = self.feature_extractor(current_states)
            preds = self.reward_classifier(th.cat([features, actions], dim=-1))
        return np.log(preds.numpy()+self.eps)

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
                nominal_preds, mean_n, logstd_n = self.forward(current_states   = nominal_data['current_states'][batch_indices, ...],
                                                                next_states      = nominal_data['next_states'][batch_indices, ...],
                                                                actions          = nominal_data['actions'][batch_indices, ...],
                                                                log_action_probs = nominal_data['log_action_probs'][batch_indices, ...])
                expert_preds, mean_e, logstd_e = self.forward(current_states    = expert_data['current_states'][batch_indices, ...],
                                                              next_states       = expert_data['next_states'][batch_indices, ...],
                                                              actions           = expert_data['actions'][batch_indices, ...],
                                                              log_action_probs  = expert_data['log_action_probs'][batch_indices, ...])

                # Reward Classifier Loss
                reward_loss = self._loss_fn(nominal_preds, nominal=True) + self._loss_fn(expert_preds, nominal=False)
                kl_loss = self._kl_loss(th.cat([mean_n, mean_e], dim=0), th.cat([logstd_n, logstd_e], dim=0))
                loss = reward_loss + self.beta*kl_loss

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.beta_update(kl_loss)

        disc_metrics =  {"discriminator/disc_loss": loss.item(),
                         "discriminator/reward_loss": reward_loss.item(),
                         "discriminator/domain_loss": domain_loss.item(),
                         "discriminator/mean_nominal_preds": nominal_r_preds.mean().item(),
                         "discriminator/max_nominal_preds": nominal_r_preds.max().item(),
                         "discriminator/min_nominal_preds": nominal_r_preds.min().item(),
                         "discriminator/mean_expert_preds": expert_r_preds.mean().item(),
                         "discriminator/max_expert_preds": expert_r_preds.max().item(),
                         "discriminator/min_expert_preds": expert_r_preds.min().item(),
                       }
        return disc_metrics





