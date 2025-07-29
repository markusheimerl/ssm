# ssm
A state space model implementation with input-dependent state transition matrix

Consider a state space model operating on sequential inputs of shape (seq_len × batch_size × input_dim). The architecture maintains internal states that evolve through learned temporal dynamics via an MLP-controlled state transition matrix, with nonlinearity applied in both the MLP and output projection. The forward propagation now follows:

$$
\begin{align*}
Z_t &= X_t W_1 \\
U_t &= \sigma(Z_t \odot \sigma(Z_t)) \\
A_t &= \tanh(U_t W_2) \\ 
H_t &= X_t B^T + H_{t-1} A_t^T \\
O_t &= H_t \odot \sigma(H_t) \\
Y_t &= O_t C^T + X_t D^T
\end{align*}
$$

The key innovation is that the state transition matrix $A_t$ is now input-dependent, computed through a multi-layer perceptron (MLP). The MLP first transforms the input $X_t$ through matrix $W_1$ to produce $Z_t$, then applies a gated activation $U_t = \sigma(Z_t \odot \sigma(Z_t))$, and finally projects through $W_2$ with tanh activation to produce the state transition matrix $A_t$. This allows the model to adapt its temporal dynamics based on the current input, while the tanh activation ensures the spectral radius of $A_t$ remains controlled for stability.

The input matrix $B$ maps current inputs to state updates, output matrix $C$ projects nonlinearly activated states to outputs, and feedthrough matrix $D$ provides direct input-output connections. The linear state evolution $H_t = X_tB^T + H_{t-1}A_t^T$ now uses the input-dependent $A_t$ matrix, enabling adaptive temporal modeling based on input context.

For gradient computation through time, we apply backpropagation through time (BPTT), where $\odot$ denotes elementwise multiplication. The gradients now include the MLP parameters:

$$
\begin{align*}
\frac{\partial L}{\partial Y_t} &= Y_t - Y_{t,\text{true}} \\
\frac{\partial L}{\partial C} &= \sum_t (\frac{\partial L}{\partial Y_t})^T O_t \\
\frac{\partial L}{\partial D} &= \sum_t (\frac{\partial L}{\partial Y_t})^T X_t \\
\frac{\partial L}{\partial O_t} &= (\frac{\partial L}{\partial Y_t})C \\
\frac{\partial L}{\partial H_t} &= \frac{\partial L}{\partial O_t} \odot [\sigma(H_t) + H_t\sigma(H_t)(1-\sigma(H_t))] + (\frac{\partial L}{\partial H_{t+1}})A_t \\
\frac{\partial L}{\partial A_t} &= (\frac{\partial L}{\partial H_t})^T H_{t-1} \\
\frac{\partial L}{\partial B} &= \sum_t (\frac{\partial L}{\partial H_t})^T X_t \\
\frac{\partial L}{\partial W_2} &= \sum_t U_t^T (\frac{\partial L}{\partial A_t} \odot (1-A_t^2)) \\
\frac{\partial L}{\partial W_1} &= \sum_t X_t^T (\frac{\partial L}{\partial Z_t})
\end{align*}
$$

The AdamW optimizer maintains exponential moving averages of gradients and their squares through $\beta_1$ and $\beta_2$, while simultaneously applying L2 regularization through weight decay $\lambda$. The learning rate is denoted by $\eta$, $t$ is the current training iteration, and $\epsilon$ is a small constant for numerical stability. For each weight matrix $W$ (including the new MLP weights $W_1$ and $W_2$), the update rule is:

$$
\begin{align*}
m &= \beta_1m + (1-\beta_1)(\frac{\partial L}{\partial W}) \\
v &= \beta_2v + (1-\beta_2)(\frac{\partial L}{\partial W})^2 \\
W &= (1-\lambda\eta)W - \eta\cdot\frac{m}{1-\beta_1^t}/\sqrt{\frac{v}{1-\beta_2^t} + \epsilon}
\end{align*}
$$

The implementation processes sequences through time-major matrix operations, where each timestep processes all batch sequences simultaneously via efficient BLAS operations. Each sequence evolves temporally as $H_0 \rightarrow H_1 \rightarrow \cdots \rightarrow H_{T-1}$ through input-dependent dynamics controlled by the MLP-generated $A_t$ matrices, while maintaining expressiveness through nonlinear output projections. The use of input-dependent state transition matrices allows the model to adapt its temporal behavior based on the current input context, potentially leading to more flexible and powerful sequence modeling.

The implementation leverages BLAS for matrix operations, enabling efficient computation on modern hardware. The MLP submodule (@markusheimerl/mlp) provides the neural network components needed for computing the input-dependent state transition matrices.

## How to run
```
sudo apt update
sudo apt install clang time libopenblas-dev
make run
```