"""Factories that plug KalMamba and the VRKN into the CoRAL experiment code, mirroring rssm_factory and the
rssm objectives.

KalMamba defaults follow Table 2 of the KalMamba paper; where the paper is silent, and for the VRKN, they
follow the VRKN code (TransitionFactory in pbecker93/vrkn).
"""
from typing import Union

import torch

import ssm_rl.common.dense_nets as dn
import ssm_rl.model_learning.objective_factories as obj_fac
from ssm_rl.kalmamba.objective import SmoothedKLObjective, SmoothingSSMObjective
from ssm_rl.kalmamba.transition import KalMambaTM
from ssm_rl.kalmamba.vrkn import VRKNTM
from ssm_rl.model_learning.abstract_objective_factory import AbstractModelObjectiveFactory
from ssm_rl.rssm.rssm import RSSM
from ssm_rl.ssm_interface.encoder_decoder import Decoder
from ssm_rl.util.config_dict import ConfigDict

nn = torch.nn


class LinearGaussianSSM(RSSM):
    """The RSSM wrapper only forwards to its transition model, so KalMamba and the VRKN reuse it."""


class VRKNTransitionFactory:

    @staticmethod
    def get_default_config(finalize_adding: bool = False) -> ConfigDict:
        config = ConfigDict()
        config.lod = 230
        config.mc_drop_prob = 0.1
        config.activation = "ReLU"
        config.hidden_size = 300
        config.init_ev = 0.9
        config.min_ev = 0.1
        config.max_ev = 0.99
        config.init_trans_var = 0.01
        config.min_trans_var = 0.001
        config.max_trans_var = 0.1
        config.init_obs_var = 1.0
        config.min_obs_var = 0.01
        config.initial_state_var = 1.0
        config.min_initial_state_var = 0.001
        config.learn_initial_state_var = True
        if finalize_adding:
            config.finalize_adding()
        return config

    @staticmethod
    def build(config: ConfigDict, obs_sizes: list[int], action_dim: int) -> VRKNTM:
        return VRKNTM(obs_sizes=obs_sizes, action_dim=action_dim, **config.get_raw_dict())


class KalMambaTransitionFactory:

    @staticmethod
    def get_default_config(finalize_adding: bool = False) -> ConfigDict:
        config = VRKNTransitionFactory.get_default_config(finalize_adding=False)
        config.activation = "SiLU"
        config.min_ev = 0.4  # Section 4.1
        # Mamba backbone (Table 2)
        config.pre_num_layers = 2
        config.pre_layer_size = 256
        config.num_blocks = 2
        config.d_model = 256
        config.d_state = 64
        config.d_conv = 2
        config.expand = 2
        # Acting (not in the paper). 0: run Mamba and the filter in step mode over the whole episode.
        # L > 0: at each step, filter the last L steps with the training-time parallel pass, so the beliefs
        # the policy acts on come from the same distribution as the training windows SAC learns from.
        config.act_window = 0
        if finalize_adding:
            config.finalize_adding()
        return config

    @staticmethod
    def build(config: ConfigDict, obs_sizes: list[int], action_dim: int) -> KalMambaTM:
        return KalMambaTM(obs_sizes=obs_sizes, action_dim=action_dim, **config.get_raw_dict())


class LinearGaussianSSMFactory:

    def __init__(self, encoder_factories, transition_factory):
        self.encoder_factories = encoder_factories
        self._transition_factory = transition_factory

    def get_default_config(self, finalize_adding: bool = True) -> ConfigDict:
        config = ConfigDict()
        for i, factory in enumerate(self.encoder_factories):
            config.add_subconf(name="encoder{}".format(i),
                               sub_conf=factory.get_default_config(finalize_adding=finalize_adding))
        config.add_subconf(name="transition",
                           sub_conf=self._transition_factory.get_default_config(finalize_adding=finalize_adding))
        config.add_subconf(name="reward_decoder", sub_conf=ConfigDict())
        config.reward_decoder.num_layers = 2
        config.reward_decoder.layer_size = 256
        config.reward_decoder.activation = "ELU"
        if finalize_adding:
            config.finalize_adding()
        return config

    def build(self,
              config: ConfigDict,
              input_sizes: list[Union[int, tuple[int, int, int]]],
              input_types: list[str],
              bypass_mask: list[bool],
              action_dim: int,
              with_obs_valid: bool):
        assert not any(bypass_mask), "Bypassing observations is not supported"
        encoders, enc_out_sizes = [], []
        for i, (input_size, input_type, factory) in enumerate(zip(input_sizes, input_types,
                                                                  self.encoder_factories)):
            enc, enc_out_size = factory.build(input_size=input_size, input_type=input_type,
                                              config=getattr(config, "encoder{}".format(i)))
            encoders.append(enc)
            enc_out_sizes.append(enc_out_size)

        transition_model = self._transition_factory.build(config.transition, obs_sizes=enc_out_sizes,
                                                          action_dim=action_dim)
        reward_layers, last_size = dn.build_layers(in_features=transition_model.feature_size,
                                                   layer_sizes=[config.reward_decoder.layer_size] *
                                                               config.reward_decoder.num_layers,
                                                   activation=config.reward_decoder.activation)
        reward_layers.append(nn.Linear(last_size, 1))
        return LinearGaussianSSM(encoders=nn.ModuleList(encoders),
                                 transition_model=transition_model,
                                 reward_decoder=Decoder(base_module=nn.Sequential(*reward_layers)))


class SmoothedKLObjectiveFactory(obj_fac.AbstractKLObjectiveFactory):

    @staticmethod
    def get_default_config(finalize_adding: bool = True) -> ConfigDict:
        config = obj_fac.AbstractKLObjectiveFactory.get_default_config(finalize_adding=False)
        config.balanced = True
        config.alpha = 0.5
        config.initial_scale_factor = 0.0
        if finalize_adding:
            config.finalize_adding()
        return config

    @staticmethod
    def build(config: ConfigDict) -> SmoothedKLObjective:
        return SmoothedKLObjective(initial_scale_factor=config.initial_scale_factor,
                                   **SmoothedKLObjectiveFactory.config_to_kwargs(config))


class SmoothingSSMObjectiveFactory(AbstractModelObjectiveFactory):

    def get_default_config(self, finalize_adding: bool = True) -> ConfigDict:
        config = super().get_default_config(finalize_adding=False)
        config.add_subconf("kl", SmoothedKLObjectiveFactory.get_default_config())
        config.regularization_scale = 1.0
        config.regularization_reduction = "sum"
        config.regularization_detach_filter = True
        if finalize_adding:
            config.finalize_adding()
        return config

    def build(self,
              model,
              target_sizes: list[Union[int, tuple[int, int, int]]],
              target_types: list[str],
              bypass_mask: list[bool],
              config: ConfigDict,
              device):
        obs_objectives, reward_objective, inverse_dyn_objective = super()._build(model=model,
                                                                                 target_sizes=target_sizes,
                                                                                 target_types=target_types,
                                                                                 bypass_mask=bypass_mask,
                                                                                 config=config,
                                                                                 device=device)
        return SmoothingSSMObjective(regularization_scale=config.regularization_scale,
                                     regularization_reduction=config.regularization_reduction,
                                     regularization_detach_filter=config.regularization_detach_filter,
                                     model=model,
                                     obs_objectives=obs_objectives,
                                     reward_objective=reward_objective,
                                     kl_objective=SmoothedKLObjectiveFactory.build(config.kl).to(device),
                                     inverse_dyn_objective=inverse_dyn_objective)
