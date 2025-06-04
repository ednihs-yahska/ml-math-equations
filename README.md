# ml-math-equations
A project to demonstrate the "universalness" of neural networks. I want to train simple neural networks to learn some simple math equations.

The training script in `linear.py` now uses **TensorBoard** to visualize how the model output compares to the target linear equation. Every 100 epochs a plot of the true line versus the network prediction in the range ``-30`` to ``30`` is logged. Training runs for 1000 epochs by default so you can track the improvement over time.

A similar script called `quadratic.py` demonstrates learning the quadratic equation
``y = x^2 + 2x + 1``. It uses a small two-layer neural network and logs the
training progress to TensorBoard just like `linear.py`.

Another example `cyclic.py` approximates the periodic function
``y = sin(2x) + cos(5x)``.  Training now covers ``-30`` to ``30`` and the
dataset can add slight noise to each input sample. The network includes dropout
layers and uses weight decay with ``AdamW`` to reduce overfitting. During
training, 20% of the data is held out for validation and an early stopping
mechanism stops if the validation loss does not improve for a while. Plots of
the model versus the target function are logged to TensorBoard every
``epochs // 10`` epochs.

Finally, `sqrt.py` trains a neural network to approximate the square root
function ``y = sqrt(x)`` on inputs from ``0`` to ``30``. The model uses two
hidden layers and logs comparison plots to TensorBoard every 100 epochs.
