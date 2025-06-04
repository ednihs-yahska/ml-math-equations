import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class SqrtDataset(torch.utils.data.Dataset):
    """Samples ``y = sqrt(x)`` for ``x`` in a configurable range."""

    def __init__(self, num_samples: int = 1000, start: float = 0.0, end: float = 30.0) -> None:
        self.num_samples = num_samples
        self.data = np.linspace(start, end, num_samples, dtype=np.float32)
        self.labels = np.sqrt(self.data)

    def __getitem__(self, index):
        x = torch.tensor(self.data[index], dtype=torch.float32)
        y = torch.tensor(self.labels[index], dtype=torch.float32)
        return x, y

    def __len__(self):
        return self.num_samples


class SqrtNN(nn.Module):
    """Small fully connected network for approximating ``sqrt``."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


def train_sqrt_model(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    num_epochs: int = 1000,
    learning_rate: float = 0.001,
    log_interval: int = 100,
    log_dir: str = "runs/sqrt",
) -> None:
    """Train ``model`` on ``dataset`` and log results to TensorBoard."""

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    writer = SummaryWriter(log_dir=log_dir)

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
                xs = torch.linspace(0.0, 30.0, steps=200).view(-1, 1)
                ys_true = torch.sqrt(xs)
                ys_pred = model(xs)

            fig = plt.figure()
            plt.plot(xs.numpy(), ys_true.numpy(), label="target")
            plt.plot(xs.numpy(), ys_pred.numpy(), label="model")
            plt.legend()
            writer.add_figure("target_vs_model", fig, global_step=epoch + 1)
            plt.close(fig)

    writer.close()
    print("Training complete.\n")


def evaluate_sqrt_model(model: nn.Module, dataset: torch.utils.data.Dataset) -> np.ndarray:
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
    """Train and evaluate a model on the square root dataset."""

    dataset = SqrtDataset(num_samples=1000, start=0.0, end=30.0)
    model = SqrtNN()

    train_sqrt_model(model, dataset, num_epochs=1000, learning_rate=0.001)

    preds = evaluate_sqrt_model(model, dataset)
    for i in range(5):
        x = dataset.data[i]
        print(
            f"x={x:.2f}\ttrue y={dataset.labels[i]:.2f}\tpredicted y={preds[i]:.2f}"
        )


if __name__ == "__main__":
    main()
