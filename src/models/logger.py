import os
import yaml
from datetime import datetime

def save_autoencoder_log(
    log_dir:str,
    model_name: str,
    input_shape,
    latent_dim: int,
    optimizer: str,
    loss_fn: str,
    metrics: list,
    train_info: dict,
    notes: str = ""):

    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_path = os.path.join(log_dir, f"{model_name}_{timestamp}.yaml")

    log_data = {
        "Autoencoder_type": {
            "name": model_name,
            "input_shape": list(input_shape),
            "latent_dim": latent_dim,
            "optimizer": optimizer,
            "loss_function": loss_fn,
            "metrics": metrics,
            "train_autoencoder": train_info,
            "notes": notes
        }
    }

    with open(file_path, "w") as f:
        yaml.dump(log_data, f, sort_keys=False)

    print(f"✅ Log salvo em: {file_path}")