import torch
import os
# may need this line if there are device mismatch issues
#os.environ["CUDA_VISIBLE_DEVICES"] = ""  
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn as nn

from agent_characterization.analysis.analyze_vae import load_vae

class VAEClusterModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        # Get parameters from model_config
        config = model_config["custom_model_config"]
        vae_path = os.path.join(config["vae_path"], "encoder.pt")
        cluster_params = config["cluster_params"]

        self.action_dim = config["action_dim"]
        
        self.vae = load_vae(
            model_path=vae_path,
            obs_shape=config["obs_shape"],
            action_dim=config["action_dim"],
            latent_dim=config["latent_dim"],
            window_length=config["window_length"],
            horizon=config["vae_horizon"]
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create distribution from cluster parameters
        self.mean = torch.tensor(cluster_params['mean'], dtype=torch.float32, device=self.device)
        self.cov = torch.tensor(cluster_params['cov'], dtype=torch.float32, device=self.device)
        eye = torch.eye(self.cov.shape[0], dtype=torch.float32, device=self.device)
        self.cov = self.cov + eye * 1e-4
        self.mvn = torch.distributions.MultivariateNormal(self.mean, self.cov)

        self.z = self.mvn.sample()

        # For storing features between forward and value_function
        self._last_obs = None
        self._last_action_probs = None

    def forward(self, input_dict, state, seq_lens):
        # Preserve the observation for value_function
        self._last_obs = input_dict
        
        # Extract image from the observation
        obs = input_dict["obs"]["image"].float()
        expected_batch_size = input_dict["obs"]["image"]
        #print("expected_batch_size",expected_batch_size.shape)
        
        # Add batch dimension if needed
        if obs.dim() < 4:
            obs = obs.unsqueeze(0)

        # TODO: check this
        #print("OBS",obs.shape)
        #obs = obs.permute(0,3,1,2).contiguous()
        
        # Number of latent points to sample
        num_samples = 1
        batch_size = obs.shape[0]
        
        # Sample a new set of latent vectors each time (no caching)
        z_samples = self.mvn.sample((num_samples,))
        
        # Process each latent sample
        with torch.no_grad():
            obs_unsq = obs.unsqueeze(1)  # Add time dimension for VAE
            all_action_logits = []
            
            for z in z_samples:
                z_expanded = z.unsqueeze(0).repeat(batch_size, 1)
                #print("z_exp",z_expanded.shape)
                logits = self.vae.decode_sequence(z_expanded, obs_unsq, True)

                #print(f"logits shape from decode_sequence: {logits.shape}")
                action_logits = logits[:, 0, :self.action_dim]
                all_action_logits.append(action_logits)
                #print(f"action_logits shape: {action_logits.shape}")
            
            # Average the logits (not the actions)
            # This preserves uncertainty and allows for exploration
            stacked_logits = torch.stack(all_action_logits, dim=0)

            #print(f"stacked_logits shape: {stacked_logits.shape}")
            averaged_logits = torch.mean(stacked_logits, dim=0)
            #print(f"averaged_logits shape: {averaged_logits.shape}")

            # Store for value function
            self._last_action_probs = averaged_logits
            

            #print(f"Final returned logits shape: {averaged_logits.shape}")
            # Apply action masking if available
            if "action_mask" in input_dict["obs"]:
                action_mask = input_dict["obs"]["action_mask"]
                inf_mask = torch.clamp(torch.log(action_mask), min=-1e38)
        
                return averaged_logits + inf_mask, state
       
        return averaged_logits, state

    def value_function(self):
        # Return a dummy value (not used for this policy)
        return torch.zeros(self._last_action_probs.shape[0], device=self.device)
