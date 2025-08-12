The three PINN aim at approximating the Van Der Poll Oscillator equation.
The Van der Pol equation is a nonlinear oscillator model described by:

\[
\frac{d^2 x}{dt^2} - \mu (1 - x^2) \frac{dx}{dt} + x = 0
\]

where \(\mu > 0\) controls the nonlinearity.  
In our code, it is written as a first-order system:

\[
\begin{cases}
\frac{dx}{dt} = y \\
\frac{dy}{dt} = \mu (1 - x^2) y - x
\end{cases}
\]

The main parameters of the StackedResPINN are:
- T_max = 3.0
- mu = 4.0
- n_stacked_mf_layers = 3
- u_0=[1.0, 0.0]
- h_sf_sizes = [40, 40, 40]
- h_res_sizes = [40, 40, 40]
