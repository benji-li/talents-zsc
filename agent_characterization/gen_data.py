import torch
import numpy as np
import argparse
from torch.utils.data import Dataset
import os
import pickle


def get_policy_names_from_checkpoint(checkpoint_path):
    policies_dir = os.path.join(checkpoint_path, "policies")
    
    if os.path.isdir(policies_dir):
        policy_names = [d for d in os.listdir(policies_dir) 
                       if os.path.isdir(os.path.join(policies_dir, d))]
        if policy_names:
            print(f"Found policies in checkpoint: {policy_names}")
            return policy_names
    
    # If we couldn't find policy names from the directory structure
    print("Couldn't find policy directories in checkpoint. Using default agent names.")
    return ["agent_0", "agent_1"]

class TrajectoryDataset(Dataset):
    """Dataset that stores full episode trajectories without windowing"""
    
    def __init__(self, trajectories=None, action_dim = 27, agents=None, encoder="lossless", layout="none"):
        """
        Args:
            trajectories: List of (states, actions) tuples for each episode
            agents: List of agent names
            encoder: Observation encoder type (lossless, onehot, nopos)
        """
        self.agents = agents if agents else []
        self.samples = []
        self.encoder = encoder
        self.layout = layout

        if trajectories and agents:
            self.samples = self._process_trajectories(trajectories)
        
        # Store the shape of observations for model creation
        self.obs_shape = None
        if len(self.samples) > 0 and 'obs_sequence' in self.samples[0] and len(self.samples[0]['obs_sequence']) > 0:
            self.obs_shape = self.samples[0]['obs_sequence'][0].shape
            print("OBS SHAPE IN TRAJ ", self.obs_shape)

        self.action_dim = action_dim
    
    def append_samples(self, new_samples):
        """Append processed samples to the existing dataset"""
        original_count = len(self.samples)

        # Update self.agents with any new agents found in the samples
        new_agents = set()
        for sample in new_samples:
            new_agents.add(sample['agent'])
        
        # Update the agents list to include any new agents
        for agent in new_agents:
            if agent not in self.agents:
                print(f"Adding new agent {agent} to dataset agents list")
                self.agents.append(agent)

        self.samples.extend(new_samples)
        print(f"Added {len(new_samples)} new samples to the dataset. Total samples: {len(self.samples)}")
        
    def _process_trajectories(self, trajectories):
        """Process raw trajectories into samples without windowing"""
        samples = []
        
        for traj in trajectories:
            # Unpack the trajectory
            states, actions = traj
            
            env_config = {
                "layout": self.layout,
                "obs_encoder": self.encoder,
                "max_steps": 1200,
                "reward_shaping_factor": 0.0,
                "reward_shaping_horizon": 0,
                "restrict_capability": False,
                "use_phi": False,
                "action_level": "low",
                #"action_mask": "high-level",
                "rew_shaping_params": None
            }
            
            from burrito_rl.env_wrapper.burrito_env import BurritoRLLibWrapper
            env = BurritoRLLibWrapper(env_config)
            
            # Process the full trajectory for each agent
            for agent_idx, agent_id in enumerate(self.agents):
                #if isinstance(agent_id, str):
                #    agent_idx = int(agent_id.split('_')[1]) if agent_id.startswith('agent_') else int(agent_id)
                
                # NOTE: only enable this for bp
                # lets just get trajs for agent_idx 0
                #if agent_idx == 1 or agent_id == "agent1":
                #    continue

                obs_agent = []
                actions_agent = []
                #partner_actions = []
                
                for j in range(len(states)):
                    # Get the current state
                    state = states[j]
                    
                    # get_obs includes the ego-centric postprocess
                    all_obs = env.get_obs(state)
                    
                    # Get agent-specific observation - extract the image component
                    agent_obs = all_obs[agent_idx]["image"]
                    
                    # Store processed observation
                    obs_agent.append(agent_obs)
                    
                    # Extract actions for current agent and partner
                    action_agent = actions[j][agent_idx]
                    
                    # Get partner actions
                    #partner_idx = (agent_idx + 1) % len(self.agents)
                    #partner_action = actions[j][partner_idx]
                    
                    # Store actions
                    actions_agent.append(action_agent)
                    #partner_actions.append(partner_action)
                
                # Store the full trajectory for this agent
                samples.append({
                    'agent': agent_id,
                    'obs_sequence': obs_agent,
                    'action_sequence': actions_agent
                    #'partner_actions': partner_actions
                })
                
        return samples

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Convert to tensors
        obs_sequence = [torch.tensor(obs, dtype=torch.float32) for obs in sample['obs_sequence']]
        action_sequence = torch.tensor(sample['action_sequence'], dtype=torch.float32)
        #partner_actions = torch.tensor(sample['partner_actions'], dtype=torch.float32)
        
        return {
            'agent': sample['agent'],
            'obs_sequence': obs_sequence,
            'action_sequence': action_sequence,
            #'partner_actions': partner_actions
        }

class HumanTrajectoryDataset(Dataset):
    """Dataset for human trajectories with fixed observation and action sequences"""
    def __init__(self, trajectories=None, action_dim = 27, agents=None, encoder="lossless", layout="none"):
        """
        Args:
            trajectories: List of (states, actions) tuples for each episode
            agents: List of agent names
            encoder: Observation encoder type (lossless, onehot, nopos)
        """
        self.agents = ["0","1"]
        self.samples = []
        self.encoder = encoder
        self.layout = layout

        self.samples = self._process_trajectories(trajectories)
        
        # Store the shape of observations for model creation
        self.obs_shape = None
        if len(self.samples) > 0 and 'obs_sequence' in self.samples[0] and len(self.samples[0]['obs_sequence']) > 0:
            self.obs_shape = self.samples[0]['obs_sequence'][0].shape
            print("OBS SHAPE IN TRAJ ", self.obs_shape)

        self.action_dim = action_dim
    
    def _process_trajectories(self, trajectories):
        """Process human json-style trajectories into samples"""
        samples = []

        for traj in trajectories:
            obs_sequences = {"1": [], "2": []}
            action_sequences = {"1": [], "2": []}
            partner_sequences = {"1": [], "2": []}

            for timestep in traj:
                actions = timestep["join_action"]
                states = timestep["states"]

                for agent_id in ["1", "2"]:
                    partner_id = "2" if agent_id == "1" else "1"

                    obs = np.array(states[agent_id])
                    obs_sequences[agent_id].append(obs)

                    action_sequences[agent_id].append(actions[int(agent_id)-1])
                    partner_sequences[agent_id].append(actions[int(partner_id)-1])

            for agent_id in ["1", "2"]:
                samples.append({
                    'agent': agent_id,
                    'obs_sequence': obs_sequences[agent_id],
                    'action_sequence': action_sequences[agent_id],
                    'partner_actions': partner_sequences[agent_id]
                })

        return samples

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Convert to tensors
        obs_sequence = [torch.tensor(obs, dtype=torch.float32) for obs in sample['obs_sequence']]
        action_sequence = torch.tensor(sample['action_sequence'], dtype=torch.float32)
        partner_actions = torch.tensor(sample['partner_actions'], dtype=torch.float32)
        
        return {
            'agent': sample['agent'],
            'obs_sequence': obs_sequence,
            'action_sequence': action_sequence,
            'partner_actions': partner_actions
        }


class ChunkedTrajectoryDataset(Dataset):
    """Dataset that creates chunks of trajectories with future observations and actions for VAE training"""
    
    def __init__(self, dataset:TrajectoryDataset, window_size=100, horizon=10):
        """
        Args:
            dataset: TrajectoryDataset containing full trajectories
            window_size: Size of each trajectory window
            horizon: Number of future steps to include for prediction
        """
        self.window_size = window_size
        self.horizon = horizon
        self.samples = []
        self.agents = []

        self.action_dim = dataset.action_dim
        self.agents = dataset.agents.copy()
        print(f"Agent Types: {self.agents}")
        self.samples = self._process_dataset_padding(dataset)

        # Store the shape of observations for model creation
        self.obs_shape = None
        if len(self.samples) > 0 and len(self.samples[0]['obs_sequence']) > 0:
            self.obs_shape = self.samples[0]['obs_sequence'][0].shape

    
    def _process_dataset_padding(self, dataset, start_ind = 10):
        """
        Process dataset into chunked samples with padding in the beginning
        
        Args:
            dataset: Dataset containing trajectory samples
            min_real_samples: timestep where we start taking chunks
        """
        chunked_samples = []
        skipped_count = 0
        
        for sample in dataset.samples:
            # Get raw data
            obs_sequence = sample['obs_sequence']
            action_sequence = sample['action_sequence']
            #partner_actions = sample['partner_actions']
            agent = sample['agent']
            
            # debugging check to make sure we don't have any truncated trajs
            if len(obs_sequence) < self.window_size + self.horizon:
                skipped_count += 1
                continue

            padding_length = self.window_size - start_ind - 1

            if padding_length > 0:
                obs_shape = obs_sequence[0].shape
                # Pad with -1s
                padded_obs = [np.full(obs_shape, -1.0) for _ in range(padding_length)]
                padded_actions = np.full(padding_length, self.action_dim)
                #padded_partner = np.full(padding_length, self.action_dim)
                
                # Add to beginning
                obs_sequence = padded_obs + list(obs_sequence)
                action_sequence = np.concatenate([padded_actions, action_sequence])
                #partner_actions = np.concatenate([padded_partner, partner_actions])

            num_chunks = len(obs_sequence) - self.window_size - self.horizon + 1

            for i in range(0, num_chunks, 4):
                # Extract window of observations and actions
                window_obs = obs_sequence[i:i+self.window_size]
                window_actions = action_sequence[i:i+self.window_size]
                #window_partner_actions = partner_actions[i:i+self.window_size]
                
                # Extract future observations and actions
                future_obs = obs_sequence[i+self.window_size:i+self.window_size+self.horizon]
                future_actions = action_sequence[i+self.window_size:i+self.window_size+self.horizon]
                #future_partner_actions = partner_actions[i+self.window_size:i+self.window_size+self.horizon]
                
                chunked_samples.append({
                    'agent': agent,
                    'obs_sequence': window_obs,
                    'action_sequence': window_actions,
                    #'partner_actions': window_partner_actions,
                    'future_obs': future_obs,
                    'future_actions': future_actions
                    #'future_partner_actions': future_partner_actions
                })
        
        print(f"Created {len(chunked_samples)} chunks from {len(dataset.samples)} trajectories")
        if skipped_count > 0:
            print(f"Skipped {skipped_count} trajectories that were too short")
            
        return chunked_samples

    def append_samples(self, new_dataset):
        """Append processed samples from another dataset"""
        original_count = len(self.samples)
        
        # Process and add new samples
        new_samples = self._process_dataset_padding(new_dataset)
        self.samples.extend(new_samples)
        
        # Update agents list if needed
        for agent in new_dataset.agents:
            if agent not in self.agents:
                print(f"Adding new agent {agent} to dataset agents list")
                self.agents.append(agent)
                
        print(f"Added {len(new_samples)} new chunks to the dataset. Total samples: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Convert observations to tensors and stack them
        obs_sequence = torch.stack([torch.tensor(obs, dtype=torch.float32) for obs in sample['obs_sequence']])
        
        # Convert actions to tensors
        action_sequence = torch.tensor(sample['action_sequence'], dtype=torch.float32)
        #partner_actions = torch.tensor(sample['partner_actions'], dtype=torch.float32)
        
        # Convert future observations to tensors and stack them
        future_obs = torch.stack([torch.tensor(obs, dtype=torch.float32) for obs in sample['future_obs']])
        
        # Convert future actions to tensors
        future_actions = torch.tensor(sample['future_actions'], dtype=torch.float32)
        #future_partner_actions = torch.tensor(sample['future_partner_actions'], dtype=torch.float32)
        
        return {
            'agent': sample['agent'],
            'obs_sequence': obs_sequence,            # Shape: [window_size, channels, height, width]
            'action_sequence': action_sequence,      # Shape: [window_size]
            #'partner_actions': partner_actions,      # Shape: [window_size]
            'future_obs': future_obs,                # Shape: [horizon, channels, height, width]
            'future_actions': future_actions,        # Shape: [horizon]
            #'future_partner_actions': future_partner_actions  # Shape: [horizon]
        }



def save_dataset(dataset, filepath):
    """Save the dataset to a file using pickle"""
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        
    with open(filepath, 'wb') as f:
        pickle.dump({
            'samples': dataset.samples,
            'agents': dataset.agents,
            'encoder': dataset.encoder
        }, f)
    print(f"Dataset saved to {filepath} with {len(dataset)} samples")


def load_dataset(filepath):
    """Load a dataset from a file"""
    if not os.path.exists(filepath):
        raise ValueError(f"Dataset file {filepath} does not exist")
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # Create a dataset with the saved encoder type if available
    encoder = data.get('encoder', 'lossless')
    dataset = TrajectoryDataset(agents=data['agents'], encoder=encoder)
    dataset.samples = data['samples']
    
    if len(dataset.samples) > 0 and 'obs_sequence' in dataset.samples[0] and len(dataset.samples[0]['obs_sequence']) > 0:
        dataset.obs_shape = dataset.samples[0]['obs_sequence'][0].shape
        
    print(f"Loaded dataset from {filepath} with {len(dataset)} samples")
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Generate trajectories for the burrito environment")
    parser.add_argument("--layout", type=str, default="open", help="Layout for the Burrito environment")
    parser.add_argument("--dataset-path", type=str, default="./data/burrito_open_bp_11pol.pkl", help="Path to save/load the dataset")
    #parser.add_argument('--model-path', type=str, default="./overcooked/src/burrito_rl/policy_params/test-2/checkpoint_000489", help="Path to the model checkpoint")
    parser.add_argument('--eval-episodes', type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument('--encoder', type=str, default="lossless", help="Observation encoder type (lossless, onehot, nopos)")
    parser.add_argument('--config', type=str, default="overcooked/src/burrito_rl/config/burrito_2p_gendata", help="Config name")
    parser.add_argument('--mode', type=str, default="eval")
    parser.add_argument('--name', type=str, default="gentraj")
    parser.add_argument('--render', default=False, type=bool, help="render during evaluation")
    parser.add_argument('--save_render', default=False, type=bool, help='save renders during evaluation.')
    parser.add_argument('--num-agents', default=2, type=int)
    parser.add_argument('--agent-0', default="group3-7")
    parser.add_argument('--agent-1', default="agent1")
    parser.add_argument('--agent-2', default=None)
    parser.add_argument('--save-vid', action='store_true', default=False, help="whether to render and save videos")
    parser.add_argument('--save-data', action='store_true', default=False, help="whether to render and save videos")
    
    args = parser.parse_args()

#    if args.agent_0 is None or args.agent_1 is None:
#        agent_names = get_policy_names_from_checkpoint(args.checkpoint)
#        print(f"Found agent names: {agent_names}")
#    else:
#        agent_names = [args.agent0,args.agent1]
#
#    args.agent_0 =

    #for idx, agent in enumerate(args.pretrained_model_path):
    #    args.pretrained_model_path[idx] = os.path.join(agent,"policy_state.pkl")

    from burrito_rl.infrastructure.train import train

    print(args.save_data,"<- saving data")
    results = train(args)

    # Create trajectories from the evaluation results
    trajectories = []
    for idx in range(args.eval_episodes):
        trajectories.append((results["ep_states"][idx], results["ep_actions"][idx]))

    action_dim = results["ep_infos"][0][0]["agent_infos"][0]["action_probs"].shape[1]
    if args.num_agents == 2:
        agent_names = [args.agent_0, args.agent_1]
    elif args.num_agents == 3:
        agent_names = [args.agent_0, args.agent_1, args.agent_2]
    else:
        raise ValueError("num_agents must be 2 or 3")
    print(f"Num episodes: {len(trajectories)}, num agents list: {agent_names}")
    print(f"Example actions[0]: {results['ep_actions'][0]}")


    # Create dataset with the appropriate agent names and encoder
    dataset = TrajectoryDataset(trajectories, action_dim, agent_names, args.encoder, args.layout)
    
    print(f"Created dataset with {dataset.__len__()} samples")
    if len(dataset) > 0:
        print(f"Each trajectory has {len(dataset.__getitem__(0)['action_sequence'])} timesteps")
        print(f"sample 1: {dataset.__getitem__(0)['agent']}")
        print(f"sample 2: {dataset.__getitem__(1)['agent']}")
        #print(f"sample 3: {dataset.__getitem__(2)['agent']}")
    
    if len(dataset) == 0:
        print("Error: No valid samples in dataset.")
        return
    
    # Get observation shape
    obs_shape = dataset.obs_shape
    #action_dim = len(list(results["ep_actions"][0][0].values())[0]) if isinstance(results["ep_actions"][0][0], dict) else None

    print(f"Observation shape: {obs_shape}, Action dim: {action_dim}")
    print(dataset.__getitem__)
    
    # Save dataset option
    #save_option = input("Save dataset? (y/n): ").lower()
    if args.save_data:
        # Check if dataset already exists
        if os.path.exists(args.dataset_path):
            print(f"Existing dataset found at {args.dataset_path}")
            existing_dataset = load_dataset(args.dataset_path)
            if existing_dataset is not None:
                # Append new dataset samples to the existing dataset
                print("Appending to existing dataset...")
                existing_dataset.append_samples(dataset.samples)
                dataset = existing_dataset
                print(f"Combined dataset now contains {len(dataset)} samples")
        
        # Save the dataset (either the combined one or just the new one)
        save_dataset(dataset, args.dataset_path)
        print(f"Dataset saved to {args.dataset_path}")
            
if __name__ == "__main__":
    main()