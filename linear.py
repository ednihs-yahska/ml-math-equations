import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class LinearEquationDataset(torch.utils.data.Dataset):
    """Simple dataset returning samples from the equation ``y = 2x + 3``."""

    def __init__(self, num_samples: int = 1000) -> None:
        self.num_samples = num_samples
        # Generate inputs uniformly spaced between -10 and 10
        self.data = np.linspace(-10, 10, num_samples, dtype=np.float32)
        # Ground truth values for the linear equation
        self.labels = -2 * self.data - 3


    def __getitem__(self, index):
        x = torch.tensor(self.data[index], dtype=torch.float32)
        y = torch.tensor(self.labels[index], dtype=torch.float32)
        return x, y

    def __len__(self):
        return self.num_samples

class LinearEquationNN(nn.Module):
    """Minimal network with a single linear layer."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

def train_linear_model(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    num_epochs: int = 10,
    learning_rate: float = 0.01,
    log_interval: int = 1,
    log_dir: str = "runs/linear",
) -> None:
    """Train ``model`` on ``dataset`` using mean squared error and log results to TensorBoard."""

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(num_epochs):
        for x, y in dataloader:
            x = x.view(-1, 1)  # Reshape for the linear layer
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y.view(-1, 1))
            loss.backward()
            optimizer.step()

        if (epoch + 1) % log_interval == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}]\tLoss: {loss.item():.6f}")

            # Generate a figure comparing the learned model to the target equation
            with torch.no_grad():
                xs = torch.linspace(-30, 30, steps=61).view(-1, 1)
                ys_true = -2 * xs - 3
                ys_pred = model(xs)

            fig = plt.figure()
            plt.plot(xs.numpy(), ys_true.numpy(), label="target")
            plt.plot(xs.numpy(), ys_pred.numpy(), label="model")
            plt.legend()
            writer.add_figure("target_vs_model", fig, global_step=epoch + 1)
            plt.close(fig)

    writer.close()
    print("Training complete.\n")

def evaluate_linear_model(model: nn.Module, dataset: torch.utils.data.Dataset) -> np.ndarray:
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
    """Train and evaluate a model on the linear equation dataset."""

    dataset = LinearEquationDataset(num_samples=1000)
    model = LinearEquationNN()

    train_linear_model(model, dataset, num_epochs=10, learning_rate=0.01)

    preds = evaluate_linear_model(model, dataset)
    weight = model.linear.weight.item()
    bias = model.linear.bias.item()
    print(f"Learned weight: {weight:.3f}")
    print(f"Learned bias: {bias:.3f}\n")

    # Display a few sample predictions
    for i in range(5):
        x = dataset.data[i]
        print(
            f"x={x:.2f}\ttrue y={dataset.labels[i]:.2f}\tpredicted y={preds[i]:.2f}"
        )


if __name__ == "__main__":
    main()
