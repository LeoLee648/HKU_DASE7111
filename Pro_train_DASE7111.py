# train.py
import os
import argparse
from pathlib import Path
from rl4co.envs.routing import TSPEnv, TSPGenerator
from rl4co.models import AttentionModelPolicy, POMO
from rl4co.utils import RL4COTrainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

def main():
    parser = argparse.ArgumentParser(description="Train TSP model with configurable hyperparameters")
    
    # 超参数
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--num_loc", type=int, default=50, help="Number of cities in TSP")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    
    args = parser.parse_args()

    # 设置随机种子（可选但推荐）
    import torch, numpy as np, random
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 构建有意义的 version 名称（关键！）
    version_name = f"epochs{args.epochs}_numloc{args.num_loc}_lr{args.lr}_bs{args.batch_size}_seed{args.seed}"
    print(f"🚀 Starting experiment: {version_name}")

    # 创建环境和模型
    generator = TSPGenerator(num_loc=args.num_loc, loc_distribution="uniform")
    env = TSPEnv(generator)
    policy = AttentionModelPolicy(env_name=env.name, num_encoder_layers=6)
    model = POMO(
        env, 
        policy, 
        batch_size=args.batch_size, 
        optimizer_kwargs={"lr": args.lr}
    )

    # Logger 和 Checkpoint
    logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name="tsp_pomo",
        version=version_name  # ← 自动创建带参数的目录！
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val/reward",
        mode="max",
        save_top_k=1,
        save_last=True,
        filename="best-{epoch}-{step}-{val/reward:.4f}",
        verbose=True,
    )

    # Trainer
    trainer = RL4COTrainer(
        max_epochs=args.epochs,
        accelerator="gpu" if args.gpu else "cpu",
        precision="16-mixed" if args.gpu else "32-true",
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=50,
    )

    # 开始训练
    trainer.fit(model)

if __name__ == "__main__":
    main()