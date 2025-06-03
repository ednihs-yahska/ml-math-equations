# ml-math-equations
A project to demonstrate the "universalness" of neural networks. I want to train simple neural networks to learn some simple math equations.

The training script in `linear.py` now uses **TensorBoard** to visualize how the model output compares to the target linear equation. Every 100 epochs a plot of the true line versus the network prediction in the range ``-30`` to ``30`` is logged. Training runs for 1000 epochs by default so you can track the improvement over time.

A similar script called `quadratic.py` demonstrates learning the quadratic equation
``y = x^2 + 2x + 1``. It uses a small two-layer neural network and logs the
training progress to TensorBoard just like `linear.py`.
