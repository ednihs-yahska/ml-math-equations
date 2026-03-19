# Universal Function Approximation with Neural Networks

The motivation for this small project is to explore what it means for a model to 'learn' something. Neural networks can be thought of as universal functions where passing some data through the network, calculating loss and backpropagating through the network, adjusts the weights in a way that given a model of appropriate structure it can learn any function.

## The role of neural networks and learning in modern day AI

Neural networks or fully connected networks such as the ones explored in this project are used in all modern AI model networks as the final stage of learning to transform the intermediate representation or hidden state to the desired output.
In the case of LLMs, transformer layers transform the input into an n-dimensional vector, which is then fed into a fully connected neural network to produce the output tokens.

This document explains how each Python program in this project demonstrates the power of neural networks as universal function approximators.

## `linear.py`: Learning a Linear Function

This script trains a neural network to learn a basic linear equation of the form `y = mx + b`. Even though a linear function is simple and could be solved directly, this test confirms the neural network's ability to approximate linear relationships and serves as a baseline.

## `quadratic.py`: Learning a Quadratic Equation

Here, the neural network learns the function `y = x^2 + 2x + 1`, a classic second-degree polynomial. This is a nonlinear function, and this example demonstrates that even shallow networks can approximate smooth curves with enough training and proper initialization.

![Quadratic](https://github.com/ednihs-yahska/ml-math-equations/blob/main/images/quadratic.gif?raw=true)

## `cyclic.py`: Learning a Periodic Function

This script trains a network to approximate the function `y = sin(2x) + cos(5x)`. Periodic functions are more complex due to their nonlinearity and infinite number of extrema. This task shows how neural networks can capture oscillatory patterns. To avoid overfitting, dropout and weight decay (L2 regularization) are used. Early stopping based on validation loss ensures generalization.

![Cyclic](https://github.com/ednihs-yahska/ml-math-equations/blob/main/images/cyclic.gif?raw=true)

## `sqrt.py`: Learning the Square Root Function

This program approximates the function `y = sqrt(x)` over the interval `[0, 30]`. The square root is a smooth but non-polynomial function, and this case illustrates how neural nets can learn non-algebraic functions with good accuracy, even when gradients near zero make learning slow for small inputs.

![Square root](https://github.com/ednihs-yahska/ml-math-equations/blob/main/images/sqrt.gif?raw=true)


## What happens if you replace python's sqrt function with a simple neural network

This graph shows the time it takes in milliseconds to calculate square root of 1.0, 25.0, 100.0, 1000.0, 10000.0 and 1e6.

| x | NumPy (µs) | Model (µs) |
|---|---|---|
| 1.0 | 0.25 | 9.56 |
| 25.0 | 0.22 | 9.69 |
| 100.0 | 0.24 | 9.37 |
| 1,000.0 | 0.24 | 9.52 |
| 10,000.0 | 0.24 | 9.24 |
| 1,000,000.0 | 0.22 | 9.30 |

![ml square root model versus square root function](https://github.com/ednihs-yahska/ml-math-equations/blob/main/images/ml_vs_function.png?raw=true)

---

## Neural Networks as Universal Approximators

The **Universal Approximation Theorem** states that a feedforward neural network with at least one hidden layer containing a finite number of neurons can approximate any continuous function on compact subsets of ℝⁿ, given sufficient capacity.

This project demonstrates this theorem in action using small, practical networks to approximate a variety of function classes: linear, polynomial, periodic, and irrational.

### References

- [Neural Networks and Deep Learning - Chapter 4](https://neuralnetworksanddeeplearning.com/chap4.html)
- [Universal Approximation Theorem - Wikipedia](https://en.wikipedia.org/wiki/Universal_approximation_theorem)
- [Theoretical Foundation for Neural Networks (Distill.pub)](https://distill.pub/2018/building-blocks/)
