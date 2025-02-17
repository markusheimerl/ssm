# ssm
A state space model implementation

Consider a nonlinear state space model operating on batched inputs of shape (batch_size × input_dim). The architecture consists of state transition matrices $A$ and $B$, output matrices $C$ and $D$, and a nonlinear swish activation, where the forward propagation follows:

$$
\begin{align*}
Z &= XB + sA \\
s' &= Z\sigma(Z) \\
Y &= s'C + XD
\end{align*}
$$

where $s$ is the current state and $s'$ is the next state. The swish activation $x\sigma(x)$ provides nonlinearity in the state transitions, yielding the following backward pass through the chain rule, where $\odot$ denotes elementwise multiplication:

$$
\begin{align*}
\frac{\partial L}{\partial Y} &= Y - Y_{\text{true}} \\
\frac{\partial L}{\partial C} &= (s')^\top(\frac{\partial L}{\partial Y}) \\
\frac{\partial L}{\partial D} &= X^\top(\frac{\partial L}{\partial Y}) \\
\frac{\partial L}{\partial s'} &= (\frac{\partial L}{\partial Y})(C)^\top \\
\frac{\partial L}{\partial Z} &= \frac{\partial L}{\partial s'} \odot [\sigma(Z) + Z\sigma(Z)(1-\sigma(Z))] \\
\frac{\partial L}{\partial A} &= s^\top(\frac{\partial L}{\partial Z}) \\
\frac{\partial L}{\partial B} &= X^\top(\frac{\partial L}{\partial Z})
\end{align*}
$$

The AdamW optimizer maintains exponential moving averages of gradients and their squares through $\beta_1$ and $\beta_2$, while simultaneously applying L2 regularization through weight decay $\lambda$. The learning rate is denoted by $\eta$, $t$ is the current training iteration, and $\epsilon$ is a small constant for numerical stability. For each matrix ($A$, $B$, $C$, $D$), the update rule is:

$$
\begin{align*}
m &= \beta_1m + (1-\beta_1)(\frac{\partial L}{\partial W}) \\
v &= \beta_2v + (1-\beta_2)(\frac{\partial L}{\partial W})^2 \\
W &= (1-\lambda\eta)W - \eta\cdot\frac{m}{1-\beta_1^t}/\sqrt{\frac{v}{1-\beta_2^t} + \epsilon}
\end{align*}
$$

The implementation leverages BLAS for matrix operations, enabling efficient computation on modern hardware. Unlike a standard MLP, the SSM maintains state between forward passes, making it particularly suitable for modeling dynamical systems and time series data. The state can be reset at the beginning of each episode to model systems with known initial conditions.

## System Stability

The stability of the learned dynamics can be analyzed through the spectral radius of the state transition matrix $A$. A system is asymptotically stable if all eigenvalues of $A$ have magnitude less than 1:

$$
\rho(A) = \max_{i} |\lambda_i| < 1
$$

where $\lambda_i$ are the eigenvalues of $A$. The implementation includes stability analysis through power iteration to estimate the spectral radius.

## How to run
```
sudo apt update
sudo apt install clang time libopenblas-dev
make run
```