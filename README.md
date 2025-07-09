# ssm
A state space model implementation

Consider a nonlinear state space model operating on sequential inputs of shape (batch_size × seq_len × input_dim). The model maintains internal states that evolve through learned temporal dynamics with Swish activation, where the forward propagation follows:

$$
\begin{align*}
z_t &= Ah_{t-1} + Bx_t \\
h_t &= z_t\sigma(z_t) \\
y_t &= Ch_t + Dx_t
\end{align*}
$$

The state transition matrix $A$ captures temporal dependencies, input matrix $B$ maps current inputs to state updates, output matrix $C$ projects states to outputs, and feedthrough matrix $D$ provides direct input-output connections. The Swish activation $z\sigma(z)$ enables nonlinear temporal modeling.

For gradient computation through time, we apply backpropagation through time (BPTT) with Swish derivative $\sigma(z) + z\sigma(z)(1-\sigma(z))$, where $\odot$ denotes elementwise multiplication:

$$
\begin{align*}
\frac{\partial L}{\partial y_t} &= y_t - y_{t,\text{true}} \\
\frac{\partial L}{\partial C} &= \sum_t h_t^\top(\frac{\partial L}{\partial y_t}) \\
\frac{\partial L}{\partial D} &= \sum_t x_t^\top(\frac{\partial L}{\partial y_t}) \\
\frac{\partial L}{\partial h_t} &= C^\top(\frac{\partial L}{\partial y_t}) + A^\top(\frac{\partial L}{\partial h_{t+1}}) \\
\frac{\partial L}{\partial z_t} &= \frac{\partial L}{\partial h_t} \odot [\sigma(z_t) + z_t\sigma(z_t)(1-\sigma(z_t))] \\
\frac{\partial L}{\partial A} &= \sum_t h_{t-1}^\top(\frac{\partial L}{\partial z_t}) \\
\frac{\partial L}{\partial B} &= \sum_t x_t^\top(\frac{\partial L}{\partial z_t})
\end{align*}
$$

The AdamW optimizer maintains exponential moving averages of gradients through $\beta_1$ and $\beta_2$, applies L2 regularization through weight decay $\lambda$, and includes gradient clipping to prevent exploding gradients during temporal backpropagation. For each parameter matrix $W$:

$$
\begin{align*}
g &= \text{clip}(\frac{\partial L}{\partial W}, \text{clip\_value}) \\
m &= \beta_1m + (1-\beta_1)g \\
v &= \beta_2v + (1-\beta_2)g^2 \\
W &= (1-\lambda\eta)W - \eta\cdot\frac{m}{1-\beta_1^t}/\sqrt{\frac{v}{1-\beta_2^t} + \epsilon}
\end{align*}
$$

The implementation processes sequences sequentially within each batch, maintaining independent state trajectories per sequence while accumulating gradients across all sequences. Each sequence evolves temporally as $h_0 \rightarrow h_1 \rightarrow \cdots \rightarrow h_{T-1}$, enabling the model to capture both immediate input-output relationships and long-term dependencies.

The implementation leverages BLAS for matrix operations, enabling efficient computation on modern hardware.

## How to run
```
sudo apt update
sudo apt install clang time libopenblas-dev
make run
```