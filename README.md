# TALENTS Adaptation

This codebase contains the full implementation and method described in ["Adaptively Coordinating with Novel Partners via Learned Latent Strategies"](https://www.arxiv.org/abs/2511.12754), presented at NeurIPS 2025.

This repo contains all the requisite code for generating a set of agent-agent trajectories, training a variational autoencoder for agent characterization, training a **T**eam **A**daptation via **L**at**E**nt **N**o-regre**T** **S**tratgies (**TALENTS**) cooperator agent with generative strategy-specific partners, and performing agent evaluation in the layouts used in the submission's experiments.

# Repository Setup
Set up a conda environment and clone the repo:
```
conda create -n talents-adaptation python=3.9.21
conda activate talents-adaptation
```
Install dependencies and submodules:
```
pip install -e .
cd overcooked
pip install -r requirements.txt
git submodule update --init --recursive
cd overcooked_ai
pip install -e .
```
## Common Issues During Setup & Fixes:
### Issue:
```
Traceback (most recent call last):
    import tree  # pip install dm_tree
ModuleNotFoundError: No module named 'tree'
```
### Fix:
`pip install dm_tree`
### Issue:
```
raise ImportError(
ImportError: ray.tune in ray > 0.7.5 requires 'tabulate'. Please re-run 'pip install ray[tune]' or 'pip install ray[rllib]'
```
### Fix:
`pip install 'ray[rllib]'`
### Issue:
`AttributeError: module 'pydantic.fields' has no attribute 'ModelField'`
### Fix:
```
pip install "pydantic<2.0.0" --force-reinstall
pip install wandb --upgrade
```

### Issue:
`OSError: [Errno 2] No such file or directory: '/data/benji/miniconda3/envs/talents-test/bin/ffmpeg' - The path specified for the ffmpeg binary might be wrong`
### Fix:
`conda install ffmpeg`

# Trajectory Aggregation
Necessary scripts for generating a trajectory dataset can be found in `agent_characterization/`, while the corresponding config files can be found in `overcooked/src/burrito_rl/config/`.
To create a dataset:
`bash agent_characterization/gen_data_new.sh`
Please note that the base agent policies for doing this are not included in the repository.

# Training an Agent Characterization Autoencoder
VAE architecture can be found in `role_encoder.py`. To train an encoder provided a trajectory dataset run the `train_encoder.py` file. Hyperparameters can be specified through command-line arguments. `train_encoder.py` will train a VAE as well as identify the clusters through K-means and silhouette analysis.

# Training a Cooperator Agent
In order to train a __TALENTS__ cooperator agent:
`bash train_clusterbr.sh`
Specify desired layout in the bash by changing the referenced config file (fc, hallway, open).
Total training timesteps can also be set in the config file, while other training parameters (including PPO and network hyperparameters) can be found in the specific config file.

# Evaluating a Trained Cooperator
Once a cooperator agent has been trained, there are two ways to evaluate it. A web app client can be used to play human-agent games, while agent-agent games can be rolled out, with the option to save the rendered game video. 
For human-agent games:
```
cd overcooked/src/burrito_rl
python -m burrito.burrito_human_evaluate
```
Specify layout and agent config in `burrito_human_evaluate.py`. Agent path and other parameters are set in the referenced config file.

For agent-agent rollouts:
Use `bash eval_br.sh` as entry point. 
