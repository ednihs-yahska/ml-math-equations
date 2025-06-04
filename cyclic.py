import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class CyclicDataset(torch.utils.data.Dataset):
    """Samples from ``y = sin(2x) + cos(5x)`` on a configurable range."""

    def __init__(
        self,
        num_samples: int = 1000,
        start: float = -30.0,
        end: float = 30.0,
        noise_std: float = 0.0,
    ) -> None:
        self.num_samples = num_samples
        self.noise_std = noise_std
        self.data = np.linspace(start, end, num_samples, dtype=np.float32)
        self.labels = np.sin(2 * self.data) + np.cos(5 * self.data)

    def __getitem__(self, index):
        x = self.data[index]
        if self.noise_std > 0.0:
            x = np.random.normal(x, self.noise_std)
        y = np.sin(2 * x) + np.cos(5 * x)
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )

    def __len__(self):
        return self.num_samples


class Sin(nn.Module):
    """Sine activation function."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return torch.sin(x)


class CyclicNN(nn.Module):
    """Fully connected network using sine activations."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            Sin(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            Sin(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


def train_cyclic_model(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    num_epochs: int = 1000,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    log_dir: str = "runs/cyclic",
    patience: int = 20,
) -> None:
    """Train ``model`` on ``dataset`` and log results to TensorBoard."""

    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    writer = SummaryWriter(log_dir=log_dir)

    log_interval = max(1, num_epochs // 10)
    best_val = float("inf")
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        for x, y in train_loader:
            x = x.view(-1, 1)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y.view(-1, 1))
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.view(-1, 1)
                outputs = model(x_val)
                vloss = criterion(outputs, y_val.view(-1, 1))
                val_loss += vloss.item() * x_val.size(0)
        val_loss /= len(val_dataset)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % log_interval == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}]\tLoss: {loss.item():.6f}\tVal Loss: {val_loss:.6f}"
            )

            with torch.no_grad():
                xs = torch.linspace(-30.0, 30.0, steps=200).view(-1, 1)
                ys_true = torch.sin(2 * xs) + torch.cos(5 * xs)
                ys_pred = model(xs)

            fig = plt.figure()
            plt.plot(xs.numpy(), ys_true.numpy(), label="target")
            plt.plot(xs.numpy(), ys_pred.numpy(), label="model")
            plt.legend()
            writer.add_figure("target_vs_model", fig, global_step=epoch + 1)
            plt.close(fig)

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    writer.close()
    print("Training complete.\n")


def evaluate_cyclic_model(model: nn.Module, dataset: torch.utils.data.Dataset) -> np.ndarray:
    """Return model predictions on ``dataset`` and print the average loss."""

    model.eval()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    criterion = nn.MSELoss()
    predictions = []
    total_loss = 0.0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.view(-1, 1)
            outputs = model(x)
            loss = criterion(outputs, y.view(-1, 1))
            total_loss += loss.item() * x.size(0)
            predictions.append(outputs.squeeze().numpy())

    avg_loss = total_loss / len(dataset)
    print(f"Average evaluation loss: {avg_loss:.6f}\n")
    return np.concatenate([p.reshape(-1) for p in predictions])


def main() -> None:
    """Train and evaluate a model on the cyclic function dataset."""

    train_dataset = CyclicDataset(num_samples=1500, noise_std=0.1)
    eval_dataset = CyclicDataset(num_samples=1000)
    model = CyclicNN()

    train_cyclic_model(model, train_dataset, num_epochs=1000, learning_rate=0.001)

    preds = evaluate_cyclic_model(model, eval_dataset)
    for i in range(5):
        x = eval_dataset.data[i]
        print(
            f"x={x:.2f}\ttrue y={eval_dataset.labels[i]:.2f}\tpredicted y={preds[i]:.2f}"
        )


if __name__ == "__main__":
    main()
