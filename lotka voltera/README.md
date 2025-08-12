The three PINNs aim at approximating the Lotka-Volterra predator-prey model. The Lotka-Volterra equations describe the interaction between two species as:

\[
\begin{cases}
\frac{dx}{dt} = \alpha x - \beta x y \\
\frac{dy}{dt} = \delta x y - \gamma y
\end{cases}
\]

where \(\alpha, \beta, \delta, \gamma > 0\) are parameters controlling the growth and interaction rates.  
In our code, these equations are used to model the population dynamics of prey \(x\) and predator \(y\).

The main parameters of the StackedResPINN are :
- T_max = 15.0s
- \alpha, \beta, \delta, \gamma = np.array([1.1, 0.4, 0.4, 0.1])
- n_stacked_mf_layers = 2
- u_0 = [2.0, 1.0]
- h_sf_sizes = [50, 50, 50]
- h_res_sizes = [50, 50, 50]
