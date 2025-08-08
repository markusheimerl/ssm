# ssm
A state space model implementation

Consider a linear state space model operating on sequential inputs of shape (seq_len × batch_size × input_dim). The architecture maintains internal states $Z_0 \rightarrow Z_1 \rightarrow \cdots \rightarrow Z_{T-1}$ that evolve through learned linear temporal dynamics, with nonlinearity applied only after the recurrent path. The forward propagation follows:

$$
\begin{align*}
Z_t &= X_tB^T + Z_{t-1}A^T \\
H_t &= Z_t\sigma(Z_t) \\
Y_t &= H_tC^T + X_tD^T
\end{align*}
$$

The state transition matrix $A$ captures temporal dependencies, input matrix $B$ maps current inputs to state updates, output matrix $C$ projects nonlinearly activated states to outputs, and feedthrough matrix $D$ provides direct input-output connections. The swish activation $Z\sigma(Z)$ interpolates between linear and nonlinear regimes, yielding the following backward pass through the chain rule, where $\odot$ denotes elementwise multiplication:

$$
\begin{align*}
\frac{\partial L}{\partial Y_t} &= Y_t - Y_{t,\text{true}} \\
\frac{\partial L}{\partial C} &= \sum_t H_t^T(\frac{\partial L}{\partial Y_t}) \\
\frac{\partial L}{\partial D} &= \sum_t X_t^T(\frac{\partial L}{\partial Y_t}) \\
\frac{\partial L}{\partial H_t} &= (\frac{\partial L}{\partial Y_t})(C^T) \\
\frac{\partial L}{\partial Z_t} &= \frac{\partial L}{\partial H_t} \odot [\sigma(Z_t) + Z_t\sigma(Z_t)(1-\sigma(Z_t))] + (\frac{\partial L}{\partial Z_{t+1}})A \\
\frac{\partial L}{\partial A} &= \sum_t (\frac{\partial L}{\partial Z_t})^T Z_{t-1} \\
\frac{\partial L}{\partial B} &= \sum_t (\frac{\partial L}{\partial Z_t})^T X_t
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

The implementation leverages BLAS for matrix operations, enabling efficient computation on modern hardware.

## How to run
```
sudo apt update
sudo apt install clang time libopenblas-dev
make run
```