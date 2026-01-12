import ray
import torch
import numpy as np
import argparse
import os
from datetime import datetime
from torch.utils.data import DataLoader
import torch.optim as optim

from agent_characterization.role_encoder import ImprovedRoleVAE
from agent_characterization.gen_data import load_dataset, ChunkedTrajectoryDataset #, UnActionsChunkedTrajectoryDataset
from agent_characterization.analysis.eval_roles import eval_and_cluster

def train_vae(args,vae, dataset, num_epochs=10, batch_size=32, learning_rate=1e-3, beta_start=0.0, beta_end=0.05, save_epochs=False):
    """Train the VAE on the dataset with validation to detect overfitting"""

    start_time = datetime.now().strftime("%Y%m%d_%H%M")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae.to(device)

    from torch.amp import autocast, GradScaler
    scaler = GradScaler() if torch.cuda.is_available() else None

    # Split dataset into training and validation sets (80/20)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)  # Shuffle the indices
    split = int(np.floor(0.8 * dataset_size))
    train_indices, val_indices = indices[:split], indices[split:]
    
    # Create DataLoader with sampler for both sets
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    
    train_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True, sampler=train_sampler)
    val_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True, sampler=val_sampler)
    
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
    
    # Tracking metrics for both training and validation
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # ----- TRAINING PHASE -----
        vae.train()
        train_total_loss = 0
        train_total_recon_loss = 0
        train_total_kl_loss = 0
        
        beta = beta_start + (beta_end - beta_start) * min(1.0, epoch / (num_epochs * 0.75))

        for batch_idx, batch in enumerate(train_dataloader):
            obs_sequence = batch['obs_sequence'].to(device)
            action_sequence = batch['action_sequence'].to(device)
            future_obs = batch['future_obs'].to(device)
            future_actions = batch['future_actions'].to(device)
            #partner_actions = batch['partner_actions'].to(device)

            if scaler is not None:
                with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                    # Forward pass
                    pred_actions, mu, log_var = vae(obs_sequence, action_sequence, future_obs)
                    loss, recon_loss, kl_loss = vae.compute_loss(pred_actions, future_actions, mu, log_var, beta)

                # Backward pass with mixed precision
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Forward pass
                pred_actions, mu, log_var = vae(obs_sequence, action_sequence, future_obs)
                loss, recon_loss, kl_loss = vae.compute_loss(pred_actions, future_actions, mu, log_var, beta)

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            train_total_loss += loss.item()
            train_total_recon_loss += recon_loss.item()
            train_total_kl_loss += kl_loss.item() 

            # Print batch stats periodically
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_dataloader)}, "
                      f"Loss: {loss.item():.4f}, Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f}")

        # Calculate average training losses for the epoch
        avg_train_loss = train_total_loss / len(train_dataloader)
        avg_train_recon_loss = train_total_recon_loss / len(train_dataloader)
        avg_train_kl_loss = train_total_kl_loss / len(train_dataloader)
        
        # ----- VALIDATION PHASE -----
        vae.eval()
        val_total_loss = 0
        val_total_recon_loss = 0
        val_total_kl_loss = 0
        
        with torch.no_grad():  # No gradients needed for validation
            for batch_idx, batch in enumerate(val_dataloader):
                obs_sequence = batch['obs_sequence'].to(device)
                action_sequence = batch['action_sequence'].to(device)
                future_obs = batch['future_obs'].to(device)
                future_actions = batch['future_actions'].to(device)
                #partner_actions = batch['partner_actions'].to(device)
                
                # Forward pass
                pred_actions, mu, log_var = vae(obs_sequence, action_sequence, future_obs)
                loss, recon_loss, kl_loss = vae.compute_loss(pred_actions, future_actions, mu, log_var, beta)
                
                val_total_loss += loss.item()
                val_total_recon_loss += recon_loss.item()
                val_total_kl_loss += kl_loss.item()
        
        avg_val_loss = val_total_loss / len(val_dataloader)
        avg_val_recon_loss = val_total_recon_loss / len(val_dataloader)
        avg_val_kl_loss = val_total_kl_loss / len(val_dataloader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}, Recon: {avg_train_recon_loss:.4f}, KL: {avg_train_kl_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Recon: {avg_val_recon_loss:.4f}, KL: {avg_val_kl_loss:.4f}")
       
        # Check for overfitting (when validation loss starts increasing while training loss decreases)
        if epoch > 0 and val_losses[-1] > val_losses[-2]:
            print(f"Validation loss increased: potential overfitting at epoch {epoch+1}")
            if train_losses[-1] < train_losses[-2]:
                print(f"  Warning: Train loss decreased while validation loss increased!")

        if (epoch+1) % 20 == 0 and save_epochs:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            upper_path = os.path.join("several_epochs", f"trial_{start_time}")
            lower_path = os.path.join(upper_path,f"{timestamp}_ep_{epoch}")
            save_path = os.path.join("epochs_training", lower_path)

            eval_and_cluster(vae, dataset, save_path, batch_size)
            save_training_summary(vae, dataset, dataset, args, save_path)
    
    return vae

def save_training_summary(vae, dataset, chunked_dataset, args, save_path):
    """
    Save a summary of the training configuration and results for reproducibility.
    
    Args:
        vae: The trained VAE model
        dataset: The original dataset before chunking
        chunked_dataset: The processed dataset used for training
        args: The command-line arguments with training configuration
        save_path: Path where to save the summary and model
    """
    import torch
    import platform
    import os
    from datetime import datetime
    
    summary_path = os.path.join(save_path, "training_summary.txt")
    
    with open(summary_path, 'w') as f:
        # Header
        f.write("=" * 50 + "\n")
        f.write("ROLE VAE TRAINING SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        # Date and time
        f.write(f"Date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # System info
        f.write("System Information:\n")
        f.write(f"- Python version: {platform.python_version()}\n")
        f.write(f"- PyTorch version: {torch.__version__}\n")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        f.write(f"- Device: {device}\n")
        if torch.cuda.is_available():
            f.write(f"- GPU: {torch.cuda.get_device_name(0)}\n")
        f.write("\n")
        
        # Training configuration
        f.write("Training Configuration:\n")
        f.write(f"- Latent dimension: {args.latent_dim}\n")
        f.write(f"- Batch size: {args.batch_size}\n")
        f.write(f"- Number of epochs: {args.num_epochs}\n")
        f.write(f"- Learning rate: {args.learning_rate}\n")
        f.write(f"- Beta VAE parameters: start={args.beta_start}, end={args.beta_end}\n")
        f.write(f"- Window length: {args.window_length}\n")
        f.write(f"- Prediction horizon: {args.horizon}\n")
        f.write("\n")
        
        # Dataset information
        f.write("Dataset Information:\n")
        f.write(f"- Original dataset path: {args.dataset_path}\n")
        f.write(f"- Original dataset size: {len(dataset.samples)} trajectories\n")
        f.write(f"- Chunked dataset size: {len(chunked_dataset)} samples\n")
        f.write(f"- Observation shape: {chunked_dataset.obs_shape}\n")
        f.write(f"- Agent types: {', '.join(chunked_dataset.agents)}\n")
        f.write("\n")
        
        # Model information
        f.write("Model Information:\n")
        f.write(f"- Model type: ImprovedRoleVAE\n")
        f.write(f"- Total parameters: {sum(p.numel() for p in vae.parameters())}\n")
        f.write(f"- Trainable parameters: {sum(p.numel() for p in vae.parameters() if p.requires_grad)}\n")
        f.write(f"- Model architecture summary:\n")
        
        # Add basic architecture details
        encoder_info = f"  Encoder: {args.window_length} time steps → {args.latent_dim} latent dims\n"
        decoder_info = f"  Decoder: Future prediction horizon: {args.horizon} steps\n"
        f.write(encoder_info)
        f.write(decoder_info)
        
        # Final path and timestamp
        f.write(f"\nModel saved to: {save_path}\n")
        f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"Training summary saved to {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate trajectories and train a windowed role encoder")
    parser.add_argument("--latent-dim", type=int, default=8, help="Dimension of the latent space")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Learning rate for training")
    parser.add_argument("--beta-start", type=float, default=0.0, help="Beta parameter for beta-VAE, start value")
    parser.add_argument("--beta-end", type=float, default=0.005, help="Beta parameter for beta-VAE, end value")
    parser.add_argument("--output", type=str, default="encoder.pt", help="Path to save the trained encoder")
    parser.add_argument("--analysis-dir", type=str, default="role_analysis", help="Directory to save role analysis")
    parser.add_argument("--dataset-path", type=str, default="../data/burrito_open_bp_11pol.pkl", help="Path to save/load the dataset")
    parser.add_argument("--window-length", type=int, default=50, help="Length of each encoder traj input")
    parser.add_argument("--horizon", type=int, default=50, help="How many actions in the future to predict")
    parser.add_argument("--save-epochs", action="store_true", default=False, help="Save the model every 5 epochs") 
    parser.add_argument("--no-obs-con", action="store_true", default=False, help="Do not use observation conditioning")
    parser.add_argument("--no-encode-actions", action="store_true", default=False, help="Use action conditioning")
    parser.add_argument("--save-dir", type=str, default="open_bp", help="Directory to save the encoder")

    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"  

    dataset = load_dataset(args.dataset_path)

    chunked_dataset = ChunkedTrajectoryDataset(
                dataset=dataset,
                window_size=args.window_length,
                horizon=args.horizon,
            )

    obs_shape = chunked_dataset.obs_shape
    action_dim = chunked_dataset.action_dim
    print(f"Chunked obs_shape: {obs_shape}, Action dim: {action_dim}")
    print(f"Created {len(chunked_dataset)} chunked samples from {len(dataset)} trajectories")

    vae = ImprovedRoleVAE(
        state_shape=obs_shape,
        action_dim=action_dim+1, # +1 to account for the padding actions
        latent_dim=args.latent_dim,
        traj_length=args.window_length,
        horizon=args.horizon,
        no_obs_con=args.no_obs_con
    )

    print("Training VAE...")
    vae = train_vae(
        args,
        vae,
        chunked_dataset,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        save_epochs=args.save_epochs
    )

    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = datetime.now().strftime("%Y%m%d")
    save_path = os.path.join(args.save_dir, f"{timestamp}_w_{args.window_length}_h_{args.horizon}_b_{args.beta_end}")
    vae_path = os.path.join(save_path,args.output)
    # Save the trained model
    os.makedirs(save_path, exist_ok=True)
    torch.save(vae.state_dict(), vae_path)
    print(f"Trained VAE saved to {vae_path}")
    
    save_training_summary(vae, dataset, chunked_dataset, args, save_path)


    print("Analyzing role representations...")

    eval_and_cluster(vae, chunked_dataset, save_path, args.batch_size)

    return
    
if __name__ == "__main__":
    main()

