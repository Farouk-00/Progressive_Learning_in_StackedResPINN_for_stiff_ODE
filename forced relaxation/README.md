The PINN aims at approximating the relaxation equation, defined as:

\[
\frac{du}{dt} = \mu \, \alpha_i \left( \cos(t) - u \right)
\]

where \(\mu > 0\) controls the relaxation speed and \(\alpha_i\) is a scaling factor.

The main parameters of the StackedResPINN are:
    T_max = 1
    \mu = 50.0
    n_stacked_mf_layers = 9
    u_0 = 0.0
    h_sf_sizes = [40, 40, 40]
    h_res_sizes = [40, 40, 40]
