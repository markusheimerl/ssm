# ssm
A state space model implementation with **selective SSM** capability

Consider a linear state space model operating on sequential inputs of shape (seq_len × batch_size × input_dim). The architecture maintains internal states that evolve through learned linear temporal dynamics, with nonlinearity applied only in the output projection. 

## Selective SSM Implementation

This implementation features **input-dependent B and C matrices** via linear projections, making the SSM "selective":

- **B_t = X_t W_B + b_B** (input-dependent input-to-state transformation)
- **C_t = X_t W_C + b_C** (input-dependent state-to-output transformation)

The selective forward propagation follows:

$$
\begin{align*}
B_t &= X_t W_B + b_B \\
C_t &= X_t W_C + b_C \\
H_t &= X_t B_t^T + H_{t-1}A^T \\
O_t &= H_t\sigma(H_t) \\
Y_t &= O_t C_t^T + X_t D^T
\end{align*}
$$

This allows the model to dynamically adapt its transformations based on the current input, providing greater expressiveness than fixed matrices.

The state transition matrix $A$ captures temporal dependencies, projection matrices $W_B, W_C$ and biases $b_B, b_C$ enable input-dependent transformations, and feedthrough matrix $D$ provides direct input-output connections. The linear state evolution with selective matrices enables both parallel computation via scan algorithms and adaptive behavior based on input context.

For gradient computation through time, we apply backpropagation through time (BPTT), where $\odot$ denotes elementwise multiplication:

$$
\begin{align*}
\frac{\partial L}{\partial Y_t} &= Y_t - Y_{t,\text{true}} \\
\frac{\partial L}{\partial C} &= \sum_t (\frac{\partial L}{\partial Y_t})^T O_t \\
\frac{\partial L}{\partial D} &= \sum_t (\frac{\partial L}{\partial Y_t})^T X_t \\
\frac{\partial L}{\partial O_t} &= (\frac{\partial L}{\partial Y_t})C \\
\frac{\partial L}{\partial H_t} &= \frac{\partial L}{\partial O_t} \odot [\sigma(H_t) + H_t\sigma(H_t)(1-\sigma(H_t))] + (\frac{\partial L}{\partial H_{t+1}})A \\
\frac{\partial L}{\partial A} &= \sum_t (\frac{\partial L}{\partial H_t})^T H_{t-1} \\
\frac{\partial L}{\partial B} &= \sum_t (\frac{\partial L}{\partial H_t})^T X_t
\end{align*}
$$

The AdamW optimizer maintains exponential moving averages of gradients and their squares through $\beta_1$ and $\beta_2$, while simultaneously applying L2 regularization through weight decay $\lambda$. The learning rate is denoted by $\eta$, $t$ is the current training iteration, and $\epsilon$ is a small constant for numerical stability. For each weight matrix $W$, the update rule is:

$$
\begin{align*}
m &= \beta_1m + (1-\beta_1)(\frac{\partial L}{\partial W}) \\
v &= \beta_2v + (1-\beta_2)(\frac{\partial L}{\partial W})^2 \\
W &= (1-\lambda\eta)W - \eta\cdot\frac{m}{1-\beta_1^t}/\sqrt{\frac{v}{1-\beta_2^t} + \epsilon}
\end{align*}
$$

The implementation processes sequences through time-major matrix operations, where each timestep processes all batch sequences simultaneously via efficient BLAS operations. Each sequence evolves temporally as $H_0 \rightarrow H_1 \rightarrow \cdots \rightarrow H_{T-1}$ through purely linear dynamics, while maintaining expressiveness through nonlinear output projections.

The implementation leverages BLAS for matrix operations, enabling efficient computation on modern hardware.

## How to run
```
sudo apt update
sudo apt install clang time libopenblas-dev
make run
```