import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class CyclicDataset(torch.utils.data.Dataset):
    """Samples from ``y = sin(2x) + cos(5x)`` on a configurable range."""

    def __init__(self, num_samples: int = 1000, start: float = -20.0, end: float = 20.0) -> None:
        self.num_samples = num_samples
        self.data = np.linspace(start, end, num_samples, dtype=np.float32)
        self.labels = np.sin(2 * self.data) + np.cos(5 * self.data)

    def __getitem__(self, index):
        x = torch.tensor(self.data[index], dtype=torch.float32)
        y = torch.tensor(self.labels[index], dtype=torch.float32)
        return x, y

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
            nn.Linear(64, 64),
            Sin(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


def train_cyclic_model(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    num_epochs: int = 1000,
    learning_rate: float = 0.001,
    log_dir: str = "runs/cyclic",
) -> None:
    """Train ``model`` on ``dataset`` and log results to TensorBoard."""

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    writer = SummaryWriter(log_dir=log_dir)

    log_interval = max(1, num_epochs // 10)

    for epoch in range(num_epochs):
        for x, y in dataloader:
            x = x.view(-1, 1)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y.view(-1, 1))
            loss.backward()
            optimizer.step()

        if (epoch + 1) % log_interval == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}]\tLoss: {loss.item():.6f}")

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

    train_dataset = CyclicDataset(num_samples=1000, start=-20.0, end=20.0)
    eval_dataset = CyclicDataset(num_samples=1000, start=-30.0, end=30.0)
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
