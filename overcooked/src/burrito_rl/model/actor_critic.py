from burrito_rl.model.model_util import _create_convolutional_layers, _create_dense_layers, _compute_layers, _preprocess_obs
import torch
import torch.nn as nn

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import FLOAT_MIN
import pdb


class ActorCritic(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        super().__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        self.model_type = kwargs.get('model_type')
        self.action_masking = kwargs.get('action_masking')
        self.critic_share_layers = kwargs.get("critic_share_layers")
        self.input_embedding_size = kwargs.get("actor_layer_sizes")[0][0]
        self._init_network(**kwargs)


    def _init_network(self, **kwargs):
        if "resnet" in str(self.model_type):
            import burrito_rl.model.resnet as resnet
            self.actor_layers = resnet.create_convolutional_layers(
                self.model_type,
                kwargs.get("input_conv_channels"),
                kwargs.get("actor_output_size")
            )
            self.critic_layers = resnet.create_convolutional_layers(
                self.model_type,
                kwargs.get("input_conv_channels"),
                1
            )
        else:
            self.actor_input_layers = _create_convolutional_layers(
                kwargs.get("input_conv_channels"),
                kwargs.get("conv_filters"),
                self.input_embedding_size
            )

            self.actor_dense_layers = _create_dense_layers(
                kwargs.get("actor_layer_sizes"),
                activation_at_end=False
            )

            self.critic_dense_layers = _create_dense_layers(
                kwargs.get("critic_layer_sizes"),
                activation_at_end=False
            )

            if not self.critic_share_layers:
                self.critic_input_layers = _create_convolutional_layers(
                    kwargs.get("input_conv_channels"),
                    kwargs.get("conv_filters"),
                    self.input_embedding_size
                )
            else:
                self.critic_input_layers = self.actor_input_layers

            self.actor_layers = nn.Sequential(
                *self.actor_input_layers,
                nn.Flatten(),
                *self.actor_dense_layers
            )
            self.critic_layers = nn.Sequential(
                *self.critic_input_layers,
                nn.Flatten(),
                *self.critic_dense_layers
            )

        self._features = None
        self._order_embedding = None

    def forward(self, input_dict, state, seq_lens):
        self._features = _preprocess_obs(self.actor_input_layers, input_dict)

        x = _compute_layers(self._features, self.actor_input_layers)
        x = x.reshape(x.shape[0], -1)
        
        self._order_embedding = input_dict['obs']['order_list'].float()
        
        if self._order_embedding.dim() < 2:
            self._order_embedding = self._order_embedding.unsqueeze(0)
        x = torch.concatenate(
            [x, self._order_embedding], dim=-1
        )
        
        x = _compute_layers(x, self.actor_dense_layers)
        if self.action_masking and "action_mask" in input_dict["obs"]:
            action_mask = input_dict["obs"]["action_mask"]
            # Convert action_mask into a [0.0 || -inf]-type mask.
            inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN/2)
            x = x + inf_mask

        return x, state

    def value_function(self, input_dict = None):
        if input_dict is not None:
            x = _preprocess_obs(self.critic_input_layers, input_dict)
            order_embedding = input_dict['obs']['order_list'].float()
        else:
            x = self._features
            order_embedding = self._order_embedding
        if order_embedding.dim() < 2:
            order_embedding = order_embedding.unsqueeze(0)

        x = _compute_layers(x, self.critic_input_layers)
        x = x.reshape(x.shape[0], -1)
        x = torch.concatenate(
            [x, order_embedding], dim=-1
        )
        x = _compute_layers(x, self.critic_dense_layers).squeeze(1)
        return x
