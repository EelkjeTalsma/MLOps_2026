import argparse
import torch
import torch.optim as optim
from ml_core.data import get_dataloaders
from ml_core.models import MLP
from ml_core.solver import Trainer
from ml_core.utils import load_config, seed_everything, setup_logger

logger = setup_logger("Experiment_Runner")

def main(args):
    # 1. Load Config & Set Seed
    # config = load_config(args.config)
    
    # 2. Setup Device
    
    # 3. Data
    # train_loader, val_loader = get_dataloaders(config)
    
    # 4. Model
    # model = MLP(...)
    
    # 5. Optimizer
    # optimizer = optim.SGD(...)
    
    # 6. Trainer & Fit
    # trainer = Trainer(...)
    # trainer.fit(...)
    config = load_config(args.config)
    seed_everything(config.get("seed", 42))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders(config)

    model = MLP(
        input_shape=config["data"]["input_shape"],
        hidden_units=config["model"]["hidden_units"],
        num_classes=config["model"].get("num_classes", 2),
        dropout_rate=config["model"].get("dropout_rate", 0.2)
    )

    optimizer = optim.SGD(model.parameters(), lr=config["training"].get("learning_rate"))

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        config=config,
        device=device,
    )
    trainer.fit(train_loader, val_loader)
    trainer.tracker.close()
    logger.info("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Simple MLP on PCAM")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()

    main(args)
    print("Skeleton: Implement main logic first.")
