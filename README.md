# ssm
A state space model implementation

Consider a linear state space model operating on sequential inputs of shape (seq_len × batch_size × input_dim). The architecture maintains internal states that evolve through learned linear temporal dynamics, with nonlinearity applied to individual components before combination. The forward propagation follows:

$$
\begin{align*}
H_t &= \sigma(X_tB^T) \odot X_tB^T + \sigma(H_{t-1}A^T) \odot H_{t-1}A^T \\
Y_t &= H_tC^T + X_tD^T
\end{align*}
$$

The state transition matrix $A$ captures temporal dependencies, input matrix $B$ maps current inputs to state updates, output matrix $C$ projects activated states to outputs, and feedthrough matrix $D$ provides direct input-output connections. The nonlinear activation $\sigma(x) \odot x$ is applied independently to input and recurrent components before summation, enabling more expressive state dynamics while maintaining computational efficiency.

For gradient computation through time, we apply backpropagation through time (BPTT), where $\odot$ denotes elementwise multiplication:

$$
\begin{align*}
\frac{\partial L}{\partial Y_t} &= Y_t - Y_{t,\text{true}} \\
\frac{\partial L}{\partial C} &= \sum_t (\frac{\partial L}{\partial Y_t})^T H_t \\
\frac{\partial L}{\partial D} &= \sum_t (\frac{\partial L}{\partial Y_t})^T X_t \\
\frac{\partial L}{\partial H_t} &= (\frac{\partial L}{\partial Y_t})C + (\frac{\partial L}{\partial H_{t+1}}) \odot \frac{\partial H_{t+1}}{\partial H_t} \\
\frac{\partial L}{\partial A} &= \sum_t (\frac{\partial L}{\partial H_t}) \odot [\sigma(H_{t-1}A^T) + H_{t-1}A^T\sigma'(H_{t-1}A^T)]^T H_{t-1} \\
\frac{\partial L}{\partial B} &= \sum_t (\frac{\partial L}{\partial H_t}) \odot [\sigma(X_tB^T) + X_tB^T\sigma'(X_tB^T)]^T X_t
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

The implementation processes sequences through time-major matrix operations, where each timestep processes all batch sequences simultaneously via efficient BLAS operations. Each sequence evolves temporally as $H_0 \rightarrow H_1 \rightarrow \cdots \rightarrow H_{T-1}$ through nonlinear dynamics applied to individual input and recurrent components, enabling efficient learning of complex temporal patterns while maintaining expressiveness through componentwise activation functions.

The implementation leverages BLAS for matrix operations, enabling efficient computation on modern hardware.

## How to run
```
sudo apt update
sudo apt install clang time libopenblas-dev
make run
```