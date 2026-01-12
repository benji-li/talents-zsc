import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNFeatureExtractor(nn.Module):
    """CNN to extract features from lossless state encodings"""
    def __init__(self, input_channels=26):
        super().__init__()
        # Using the architecture you specified
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        
    def forward(self, x):
        # x shape: (batch_size, channels, width, height)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten the spatial dimensions only, keeping batch dimension
        return x.view(x.size(0), -1)

class ImprovedRoleVAE(nn.Module):
    def __init__(self, state_shape, action_dim, latent_dim, traj_length, horizon=10, no_obs_con=False):
        """
        Args:
            state_shape: Tuple (channels, height, width) of single observation
            action_dim: Dimension of action space (number of possible actions)
            latent_dim: Dimension of latent space
            traj_length: Length of past trajectory window
            horizon: Number of timesteps forward in time to predict
        """
        super().__init__()
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.horizon = horizon
        self.traj_length = traj_length
        self.no_obs_con = no_obs_con
        
        # CNN feature extractor (shared between encoder and decoder)
        self.feature_extractor = CNNFeatureExtractor(input_channels=state_shape[0])
        
        # Forward pass through a sample input to determine feature size
        dummy_input = torch.zeros(1, *state_shape)
        with torch.no_grad():
            self.cnn_feature_size = self.feature_extractor(dummy_input).shape[1]
        
        #print(f"CNN feature size: {self.cnn_feature_size}")
        
        # Action embedding (to convert discrete action integers to vectors)
        self.action_embedding_dim = 8  # Dimension for action embeddings
        self.action_embedding = nn.Embedding(action_dim, self.action_embedding_dim)
        
        # Temporal feature integration with action inputs
        self.temporal_encoder = nn.GRU(
            input_size=self.cnn_feature_size + self.action_embedding_dim,  # CNN features + action embedding
            hidden_size=256,  # as specified
            batch_first=True
        )
        
        # Encoder hidden layers 
        self.encoder = nn.Sequential(
            nn.Linear(256, 256),  # hidden layer size 256
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2)
        )
        
        # Decoder with recurrent layer to process future observations
        if self.no_obs_con:
            self.decoder_rnn = nn.GRU(
                input_size=latent_dim , #+ self.cnn_feature_size,  # latent vector + future obs features
                hidden_size=256,  # recurrent layer size 256
                batch_first=True
            )
        else:
            self.decoder_rnn = nn.GRU(
                input_size=latent_dim + self.cnn_feature_size,  # latent vector + future obs features
                hidden_size=256,  # recurrent layer size 256
                batch_first=True
            )
 
        # Action predictor from decoder hidden state
        self.action_predictor = nn.Linear(256, action_dim)

    def encode(self, obs_sequence, action_sequence):
        """
        Encode a sequence of observations and actions into a latent vector
        
        Args:
            obs_sequence: Tensor of shape (batch_size, traj_length, channels, height, width)
            action_sequence: Tensor of shape (batch_size, traj_length)
            
        Returns:
            z: Latent vector
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
        """
        batch_size = obs_sequence.size(0)
        
        # Process each timestep with CNN and combine with actions
        features = []
        for t in range(self.traj_length):
            # Extract observation at time t
            obs_t = obs_sequence[:, t]  # Shape: (batch_size, channels, height, width)

            # fix device issue running on gretel
            obs_t = obs_t.to(next(self.parameters()).device)
 
            # Extract features with CNN
            cnn_feat_t = self.feature_extractor(obs_t)  # Shape: (batch_size, cnn_feature_size)
            
            # Get action at time t and embed it
            action_t = action_sequence[:, t].long()  # Convert to long for embedding
            action_embedding_t = self.action_embedding(action_t)  # Shape: (batch_size, action_embedding_dim)
            
            # Combine CNN features with action embedding
            combined_feat_t = torch.cat([cnn_feat_t, action_embedding_t], dim=1)
            features.append(combined_feat_t)
        
        # Stack features along time dimension
        features = torch.stack(features, dim=1)  # Shape: (batch_size, traj_length, cnn_feature_size+action_embedding_dim)
        
        # Process temporal features with GRU
        _, h_n = self.temporal_encoder(features)  # h_n shape: (1, batch_size, 256)
        h_n = h_n.squeeze(0)  # Shape: (batch_size, 256)
        
        # Generate latent parameters
        latent_params = self.encoder(h_n)  # Shape: (batch_size, latent_dim*2)
        mu, log_var = latent_params.chunk(2, dim=1)
        
        # Reparameterization trick
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
            
        return z, mu, log_var


    def decode_sequence(self, z, future_obs, gen_ag = False):
        """
        Decode a latent vector into a sequence of actions given future observations
        
        Args:
            z: Latent vector of shape (batch_size, latent_dim)
            future_obs: Tensor of shape (batch_size, horizon, channels, height, width)
            gen_ag: False when training VAE, True when generating agent actions to train BR
            
        Returns:
            action_sequence: Predicted actions of shape (batch_size, horizon, action_dim)
        """
        batch_size = z.size(0)
        h_0 = torch.zeros(1, batch_size, 256, device=z.device)  # Initial hidden state
        pred_actions = []

        #print(z.shape,"Z SHAPE")
        #print(future_obs.shape, "FUTURE_OBS SHAPE")

        horizon = self.horizon if not gen_ag else 1
        for t in range(horizon):
            if self.no_obs_con:
                output, h_0 = self.decoder_rnn(z.unsqueeze(1), h_0)
            else:
                obs_t = future_obs[:, t]  # Shape: (batch_size, channels, height, width)
                #print(obs_t.shape,"VAE OBS SHAPE")

                # fix device issue running on gretel
                obs_t = obs_t.to(next(self.parameters()).device)
                
                # Extract features with CNN
                obs_features = self.feature_extractor(obs_t)  # Shape: (batch_size, cnn_feature_size)
                
                # Combine latent vector with observation features
                decoder_input = torch.cat([z, obs_features], dim=1).unsqueeze(1)  # Add time dimension
                
                # Process with decoder RNN
                output, h_0 = self.decoder_rnn(decoder_input, h_0)
            
            # Predict action
            next_action_logits = self.action_predictor(output.squeeze(1))
            pred_actions.append(next_action_logits)

        # Stack predictions along time dimension
        return torch.stack(pred_actions, dim=1)  # Shape: (batch_size, horizon, action_dim)

    def forward(self, obs_sequence, action_sequence, future_obs):
        """
        Forward pass through the VAE
        
        Args:
            obs_sequence: Tensor of shape (batch_size, traj_length, channels, height, width)
            action_sequence: Tensor of shape (batch_size, traj_length) - agent's own past actions
            future_obs: Tensor of shape (batch_size, horizon, channels, height, width) - future observations
            
        Returns:
            predicted_actions: Predicted future actions of shape (batch_size, horizon, action_dim)
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
        """
        z, mu, log_var = self.encode(obs_sequence, action_sequence)
        predicted_actions = self.decode_sequence(z, future_obs)
        return predicted_actions, mu, log_var

    def compute_loss(self, pred_actions, target_actions, mu, log_var, beta=0.01):
        """
        Compute the VAE loss: reconstruction loss + beta * KL divergence
        
        Args:
            pred_actions: Predicted actions of shape (batch_size, horizon, action_dim)
            target_actions: Ground truth actions of shape (batch_size, horizon)
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
            beta: Weight for KL divergence term
            
        Returns:
            total_loss: Combined loss
            recon_loss: Reconstruction loss component
            kl_loss: KL divergence component
        """
        # Reconstruction loss (cross-entropy for each step in the horizon)
        recon_loss = 0
        for t in range(pred_actions.size(1)):
            recon_loss += F.cross_entropy(pred_actions[:, t], target_actions[:, t].long())
        recon_loss /= pred_actions.size(1)  # Average over time steps
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        kl_loss = kl_loss / mu.size(0)  # Normalize by batch size
        
        # Total loss
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
