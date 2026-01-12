#!/usr/bin/env python
# ─────────────────────────────────────────────────────────────────────────────
# 1) gRPC fork‐support OFF — must come before *any* import of grpc/Ray/W&B
import os
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
#os.environ["CUDA_VISIBLE_DEVICES"] ="0" #"5,6,7"
# 2) SIGINT short-circuit so we never hang on multiprocessing queue joins
import signal, sys
def _sigint_handler(signum, frame):
    import ray
    import wandb
    wandb.finish()
    ray.shutdown()
    os._exit(0)
signal.signal(signal.SIGINT, _sigint_handler)

if __name__ == "__main__":
    import argparse
    from burrito_rl.infrastructure.train import train

    parser = argparse.ArgumentParser()

    # training setting args
    parser.add_argument("--config", type=str, default="usar")
    parser.add_argument("--name", type=str, default=None, help="Desired name description")
    parser.add_argument("--mode", choices=["train","tune","eval"], default="train")
    parser.add_argument("--model_path", type=str, default=None, help="folder path to save or load model")
    parser.add_argument("--stop_iters", type=int, default=100, help="Number of iterations to train.")
    parser.add_argument("--timesteps_total", type=int, default=100000)
    parser.add_argument("--ckpt_freq", type=int, default=20, help="model parameter checkpoint save frequency")
    parser.add_argument("--load_model", action='store_true', default=False, help="whether load model to keep training")
    parser.add_argument("--restore_path", type=str, default=None, help="the path to restore checkpoint")
    parser.add_argument("--logger", choices=['tensorboard', 'wandb'], default='tensorboard')
    parser.add_argument("--wandb_project",type=str, default=None)

    # evaluation setting args
    parser.add_argument("--eval_episodes", default=3, type=int)
    parser.add_argument('--render', default=False, type=bool, help="render during evaluation")
    parser.add_argument('--save_render', action='store_true', default=False, help='save renders during evaluation.')
    parser.add_argument('--save_vid', action='store_true', help='save video during evaluation')
    # … your parser setup …
    args = parser.parse_args()

    # 5) Run your training
    train(args=args)
