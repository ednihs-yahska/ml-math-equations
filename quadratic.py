import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class QuadraticEquationDataset(torch.utils.data.Dataset):
    """Dataset for the equation ``y = x**2 + 2x + 1``."""

    def __init__(self, num_samples: int = 1000) -> None:
        self.num_samples = num_samples
        self.data = np.linspace(-10, 10, num_samples, dtype=np.float32)
        self.labels = self.data ** 2 + 2 * self.data + 1

    def __getitem__(self, index):
        x = torch.tensor(self.data[index], dtype=torch.float32)
        y = torch.tensor(self.labels[index], dtype=torch.float32)
        return x, y

    def __len__(self):
        return self.num_samples


class QuadraticEquationNN(nn.Module):
    """A small neural network for approximating the quadratic function."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_quadratic_model(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    num_epochs: int = 1000,
    learning_rate: float = 0.01,
    log_interval: int = 100,
    log_dir: str = "runs/quadratic",
) -> None:
    """Train ``model`` on ``dataset`` and log results to TensorBoard."""

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
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
                xs = torch.linspace(-30, 30, steps=61).view(-1, 1)
                ys_true = xs ** 2 + 2 * xs + 1
                ys_pred = model(xs)

            fig = plt.figure()
            plt.plot(xs.numpy(), ys_true.numpy(), label="target")
            plt.plot(xs.numpy(), ys_pred.numpy(), label="model")
            plt.legend()
            writer.add_figure("target_vs_model", fig, global_step=epoch + 1)
            plt.close(fig)

    writer.close()
    print("Training complete.\n")


def evaluate_quadratic_model(model: nn.Module, dataset: torch.utils.data.Dataset) -> np.ndarray:
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
    """Train and evaluate a model on the quadratic equation dataset."""

    dataset = QuadraticEquationDataset(num_samples=1000)
    model = QuadraticEquationNN()

    train_quadratic_model(model, dataset, num_epochs=1000, learning_rate=0.01)

    preds = evaluate_quadratic_model(model, dataset)

    for i in range(5):
        x = dataset.data[i]
        print(
            f"x={x:.2f}\ttrue y={dataset.labels[i]:.2f}\tpredicted y={preds[i]:.2f}"
        )


if __name__ == "__main__":
    main()
