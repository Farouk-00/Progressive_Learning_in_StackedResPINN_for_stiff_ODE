The three PINNs aim at approximating the Lotka-Volterra predator-prey model. The Lotka-Volterra equations describe the interaction between two species as:

\[
\begin{cases}
\frac{dx}{dt} = \alpha x - \beta x y \\
\frac{dy}{dt} = \delta x y - \gamma y
\end{cases}
\]

where \(\alpha, \beta, \delta, \gamma > 0\) are parameters controlling the growth and interaction rates.  
In our code, these equations are used to model the population dynamics of prey \(x\) and predator \(y\).
