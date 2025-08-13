The PINN aims at approximating the relaxation equation, defined as:

\[
\frac{du}{dt} = \mu \, \alpha_i \left( \cos(t) - u \right)
\]

where \(\mu > 0\) controls the relaxation speed and \(\alpha_i\) is a scaling factor.

The main parameters of the StackedResPINN are:
- T_max = 1
- \mu = 50.0
- n_stacked_mf_layers = 9
- u_0 = 0.0
- h_sf_sizes = [40, 40, 40]
- h_res_sizes = [40, 40, 40]

As the stiffness indicator is constant equals to -\mu, the sequence (\gamma_i) is constant equal to (i+1)/(N+1), and then there is no difference between the adaptive_gamma and constant_gamma versions of the Stacked Residual PINN.
