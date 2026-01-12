from burrito_rl.model.model_util import _create_convolutional_layers, _create_dense_layers, _compute_layers, _preprocess_obs
import torch
import torch.nn as nn

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import FLOAT_MIN
import pdb

class ActionBiasedClusterConditionedActorCritic(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        super().__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        self.model_type = kwargs.get('model_type')
        self.action_masking = kwargs.get('action_masking')
        self.critic_share_layers = kwargs.get("critic_share_layers")
        self.input_embedding_size = kwargs.get("actor_layer_sizes")[0][0]

        self.n_clusters = model_config["custom_model_config"].get("num_clusters")

        self.cluster_action_biases = nn.Parameter(
            torch.zeros(self.n_clusters, num_outputs)
        )
        self.cluster_bias_weight = nn.Parameter(torch.tensor(2.0)) # NOTE: manually set this

        self.cluster_value_biases = nn.Parameter(
            torch.zeros(self.n_clusters, 1)
        )

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
        batch_size = x.shape[0]
        
        #print(f"CNN output shape: {x.shape}")  # Check CNN output shape
        self._order_embedding = input_dict['obs']['order_list'].float()
        
        #print(f"Order embedding shape: {self._order_embedding.shape}")
        if self._order_embedding.dim() < 2:
            self._order_embedding = self._order_embedding.unsqueeze(0)

        x = torch.cat([x, self._order_embedding], dim=-1)

        action_logits = _compute_layers(x, self.actor_dense_layers)

        if "cluster_id" in input_dict["obs"]:
            cluster_id = input_dict["obs"]["cluster_id"].float()
            if cluster_id.dim() < 2:
                cluster_id = cluster_id.unsqueeze(0)
                #print(f"After unsqueeze, cluster_id shape: {cluster_id.shape}")

            self._cluster_onehot = cluster_id
            
            #print(self.cluster_action_biases, "CLUSTER ACTION BIASES")
            #print(self.cluster_action_biases.shape, "CLUSTER ACTION BIASES SHAPE")


            # apply the cluster biases
            cluster_biases = torch.matmul(cluster_id, self.cluster_action_biases)
            #print(cluster_biases, "CLUSTER BIASES")
            #print(action_logits,"ACTION LOGITS")
            action_logits = action_logits + cluster_biases * self.cluster_bias_weight
            #print(action_logits, "MODIFIED ACTION LOGITS")

        if self.action_masking and "action_mask" in input_dict["obs"]:
            action_mask = input_dict["obs"]["action_mask"]
            # Convert action_mask into a [0.0 || -inf]-type mask.
            inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN/2)
            action_logits = action_logits + inf_mask

        return action_logits, state

    def value_function(self, input_dict = None):
        if input_dict is not None:
            x = _preprocess_obs(self.critic_input_layers, input_dict)
            order_embedding = input_dict['obs']['order_list'].float()
            
            if "cluster_id" in input_dict["obs"]:
                cluster_onehot = input_dict["obs"]["cluster_id"].float()
                if cluster_onehot.dim() < 2:
                    cluster_onehot = cluster_onehot.unsqueeze(0)
            else:
                cluster_onehot = None
        else:
            x = self._features
            order_embedding = self._order_embedding
            cluster_onehot = self._cluster_onehot

        if order_embedding.dim() < 2:
            order_embedding = order_embedding.unsqueeze(0)

        x = _compute_layers(x, self.critic_input_layers)
        x = x.reshape(x.shape[0], -1)
        
        x = torch.cat([x, order_embedding], dim=-1)
        
        value_estimate = _compute_layers(x, self.critic_dense_layers).squeeze(1)

        if cluster_onehot is not None:
            value_biases = torch.matmul(cluster_onehot, self.cluster_value_biases).squeeze(-1)
            value_estimate = value_estimate + value_biases
            
        return value_estimate

class ClusterConditionedActorCritic(TorchModelV2, nn.Module):

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        super().__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        self.model_type = kwargs.get('model_type')
        self.action_masking = kwargs.get('action_masking')
        self.critic_share_layers = kwargs.get("critic_share_layers")
        self.input_embedding_size = kwargs.get("actor_layer_sizes")[0][0]

        self.n_clusters = model_config["custom_model_config"].get("num_clusters")
        self.cluster_embedding_size = 32
        self.cluster_embedding = nn.Linear(self.n_clusters, self.cluster_embedding_size)

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

    def __forward(self, input_dict, state, seq_lens):
        print("\n===== FORWARD PASS DEBUG =====")
        print(f"Input dict keys: {list(input_dict.keys())}")
        print(f"Obs keys: {list(input_dict['obs'].keys())}")
        
        self._features = _preprocess_obs(self.actor_input_layers, input_dict)
        x = _compute_layers(self._features, self.actor_input_layers)
        x = x.reshape(x.shape[0], -1)
        
        print(f"CNN output (x) shape: {x.shape}")
        
        self._order_embedding = input_dict['obs']['order_list'].float()
        print(f"Original order_embedding shape: {self._order_embedding.shape}")
        
        if self._order_embedding.dim() < 2:
            self._order_embedding = self._order_embedding.unsqueeze(0)
            print(f"After unsqueeze, order_embedding shape: {self._order_embedding.shape}")
        
        # Process cluster ID if available
        if "cluster_id" in input_dict["obs"]:
            print("Cluster ID is present in input")
            cluster_id = input_dict["obs"]["cluster_id"].float()
            print(f"Original cluster_id shape: {cluster_id.shape}")
            
            if cluster_id.dim() < 2:
                cluster_id = cluster_id.unsqueeze(0)
                print(f"After unsqueeze, cluster_id shape: {cluster_id.shape}")
            
            print(f"Cluster embedding layer weight shape: {self.cluster_embedding.weight.shape if hasattr(self.cluster_embedding, 'weight') else 'N/A'}")
            self._cluster_embedding = self.cluster_embedding(cluster_id)
            print(f"After embedding, _cluster_embedding shape: {self._cluster_embedding.shape}")
            
            # Right before concatenation:
            print(f"PRE-CONCAT SHAPES: x={x.shape}, order_embedding={self._order_embedding.shape}, cluster_embedding={self._cluster_embedding.shape}")
            
            # Concatenate all embeddings
            try:
                x = torch.cat([x, self._order_embedding, self._cluster_embedding], dim=-1)
                print(f"After concatenation, shape: {x.shape}")
            except RuntimeError as e:
                print(f"ERROR DURING CONCATENATION: {e}")
                print(f"Tensor details: x={x.dtype}, order={self._order_embedding.dtype}, cluster={self._cluster_embedding.dtype}")
                # Manually show tensor content for small batches
                if x.shape[0] < 5:
                    print(f"x content: {x}")
                    print(f"order_embedding content: {self._order_embedding}")
                    print(f"cluster_embedding content: {self._cluster_embedding}")
                raise e
        else:
            print("NO cluster ID in input")
            # Just concatenate order embedding if no cluster ID
            x = torch.cat([x, self._order_embedding], dim=-1)
            print(f"After concatenation with only order embedding, shape: {x.shape}")
        
        print(f"Expected input shape for dense layer: {self.actor_dense_layers[0].in_features if hasattr(self.actor_dense_layers[0], 'in_features') else 'N/A'}")
        print("===== END DEBUG =====\n")
        
        x = _compute_layers(x, self.actor_dense_layers)
        if self.action_masking and "action_mask" in input_dict["obs"]:
            action_mask = input_dict["obs"]["action_mask"]
            # Convert action_mask into a [0.0 || -inf]-type mask.
            inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN/2)
            x = x + inf_mask
            
        return x, state

    def forward(self, input_dict, state, seq_lens):
        self._features = _preprocess_obs(self.actor_input_layers, input_dict)

        x = _compute_layers(self._features, self.actor_input_layers)
        x = x.reshape(x.shape[0], -1)
        batch_size = x.shape[0]
        
        #print(f"CNN output shape: {x.shape}")  # Check CNN output shape
        self._order_embedding = input_dict['obs']['order_list'].float()
        
        #print(f"Order embedding shape: {self._order_embedding.shape}")
        if self._order_embedding.dim() < 2:
            self._order_embedding = self._order_embedding.unsqueeze(0)

        # Process cluster ID if available
        if "cluster_id" in input_dict["obs"]:

            #print("==============================")
            #print("Cluster ID is present")
            cluster_id = torch.zeros_like(input_dict["obs"]["cluster_id"]).float()
            #print(input_dict["obs"])
            #print(cluster_id, "CLUSTER ID")
            #cluster_id = torch.tensor([0.0,0.0,0.0,0.0])

            #print(f"Cluster ID shape: {cluster_id.shape}")
            if cluster_id.dim() < 2:
                cluster_id = cluster_id.unsqueeze(0)
            self._cluster_embedding = self.cluster_embedding(cluster_id)

            if self._cluster_embedding.shape[0] == 1 and batch_size > 1:
                print(f"WARNING: Had to expand cluster embedding from batch size 1 to {batch_size}")
                self._cluster_embedding = self._cluster_embedding.expand(batch_size, -1)

            #print(f"Cluster embedding shape: {self._cluster_embedding.shape}")
            # Concatenate all embeddings
            x = torch.cat([x, self._order_embedding, self._cluster_embedding], dim=-1)
        else:
            # Just concatenate order embedding if no cluster ID
            x = torch.cat([x, self._order_embedding], dim=-1)
        #print(f"Final concatenated shape: {x.shape}")
        #print(f"Expected input shape for dense layer: {self.actor_dense_layers[0].in_features}")
    
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

            if "cluster_id" in input_dict["obs"]:
                cluster_id = input_dict["obs"]["cluster_id"].float()
                if cluster_id.dim() < 2:
                    cluster_id = cluster_id.unsqueeze(0)
                cluster_embedding = self.cluster_embedding(cluster_id)
            else:
                cluster_embedding = None

        else:
            x = self._features
            order_embedding = self._order_embedding
            cluster_embedding = self._cluster_embedding

        if order_embedding.dim() < 2:
            order_embedding = order_embedding.unsqueeze(0)

        x = _compute_layers(x, self.critic_input_layers)
        x = x.reshape(x.shape[0], -1)

        x = torch.cat([x, order_embedding, cluster_embedding], dim=-1)

        x = _compute_layers(x, self.critic_dense_layers).squeeze(1)
        return x
