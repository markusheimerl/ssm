# ssm
A state space model implementation

Consider a linear state space model operating on sequential inputs of shape (seq_len × batch_size × input_dim). The architecture maintains internal states $H_0 \rightarrow H_1 \rightarrow \cdots \rightarrow H_{T-1}$ that evolve through learned linear temporal dynamics, with nonlinearity applied only after the recurrent path. The forward propagation follows:

$$
\begin{align*}
H_t &= X_tB^T + H_{t-1}A^T \\
S_t &= H_t\sigma(H_t) \\
Y_t &= S_tC^T + X_tD^T
\end{align*}
$$

The state transition matrix $A$ captures temporal dependencies, input matrix $B$ maps current inputs to state updates, output matrix $C$ projects nonlinearly activated states to outputs, and feedthrough matrix $D$ provides direct input-output connections. The swish activation $H\sigma(H)$ interpolates between linear and nonlinear regimes, yielding the following backward pass through the chain rule, where $\odot$ denotes elementwise multiplication:

$$
\begin{align*}
\frac{\partial L}{\partial Y_t} &= Y_t - Y_{t,\text{true}} \\
\frac{\partial L}{\partial C} &= \sum_t S_t^T(\frac{\partial L}{\partial Y_t}) \\
\frac{\partial L}{\partial D} &= \sum_t X_t^T(\frac{\partial L}{\partial Y_t}) \\
\frac{\partial L}{\partial S_t} &= (\frac{\partial L}{\partial Y_t})(C^T) \\
\frac{\partial L}{\partial H_t} &= \frac{\partial L}{\partial S_t} \odot [\sigma(H_t) + H_t\sigma(H_t)(1-\sigma(H_t))] + (\frac{\partial L}{\partial H_{t+1}})A \\
\frac{\partial L}{\partial A} &= \sum_t (\frac{\partial L}{\partial H_t})^T H_{t-1} \\
\frac{\partial L}{\partial B} &= \sum_t (\frac{\partial L}{\partial H_t})^T X_t
\end{align*}
$$

The Lion optimizer maintains exponential moving averages of gradients through $\beta_1$ and $\beta_2$, while simultaneously applying L2 regularization through weight decay $\lambda$. The learning rate is denoted by $\eta$ and $t$ is the current training iteration. For each weight matrix $W$, the update rule is:

$$
\begin{align*}
c_t &= \beta_1 m_{t-1} + (1-\beta_1)g_t \\
w_t &= w_{t-1} - \eta(\lambda w_{t-1} + \text{sign}(c_t)) \\
m_t &= \beta_2 m_{t-1} + (1-\beta_2)g_t
\end{align*}
$$

The implementation leverages BLAS for matrix operations, enabling efficient computation on modern hardware.

## How to run
```
sudo apt update
sudo apt install clang time libopenblas-dev
make run
```