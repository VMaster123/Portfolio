# AeroTwin Mathematical Foundations

This document outlines the mathematical models used in the AeroTwin digital twin system.

## 1. 6-DOF Rigid Body Dynamics (Flight Dynamics)

The UAV is modeled as a rigid body with 6 degrees of freedom. The state vector $\mathbf{x}$ is defined as:

$$\mathbf{x} = [r^T, v^T, q^T, \omega^T]^T$$

Where:
- $\mathbf{r} = [x, y, z]^T$ is the position in the North-East-Down (NED) inertial frame.
- $\mathbf{v} = [u, v, w]^T$ is the velocity in the body-fixed frame.
- $\mathbf{q} = [q_w, q_x, q_y, q_z]^T$ is the unit quaternion representing orientation.
- $\mathbf{\omega} = [p, q, r]^T$ is the angular velocity in the body-fixed frame.

### Translational Dynamics
The rate of change of position is given by rotating the body velocity into the inertial frame:
$$\dot{\mathbf{r}} = \mathbf{R}(\mathbf{q}) \mathbf{v}$$
Where $\mathbf{R}(\mathbf{q})$ is the rotation matrix derived from the quaternion.

The acceleration in the body frame follows Newton's second law:
$$m(\dot{\mathbf{v}} + \mathbf{\omega} \times \mathbf{v}) = \mathbf{F}_{gravity} + \mathbf{F}_{aero} + \mathbf{F}_{thrust}$$
$$\dot{\mathbf{v}} = \frac{1}{m}\mathbf{F}_{total} - \mathbf{\omega} \times \mathbf{v}$$

### Rotational Dynamics
The quaternion derivative is:
$$\dot{\mathbf{q}} = \frac{1}{2} \mathbf{q} \otimes \begin{bmatrix} 0 \\ \mathbf{\omega} \end{bmatrix}$$

The angular acceleration follows Euler's equations:
$$\mathbf{I}\dot{\mathbf{\omega}} + \mathbf{\omega} \times (\mathbf{I}\mathbf{\omega}) = \mathbf{M}_{aero} + \mathbf{M}_{thrust}$$
$$\dot{\mathbf{\omega}} = \mathbf{I}^{-1} (\mathbf{M}_{total} - \mathbf{\omega} \times (\mathbf{I}\mathbf{\omega}))$$

Where $\mathbf{I}$ is the inertia tensor.

## 2. Aerodynamic Modeling

Forces and moments are modeled using dimensionless coefficients:
$$L = \frac{1}{2}\rho V^2 S C_L(\alpha)$$
$$D = \frac{1}{2}\rho V^2 S C_D(\alpha)$$

Where:
- $\rho$ is air density.
- $V$ is airspeed.
- $S$ is reference area.
- $\alpha$ is angle of attack.

The inverse problem engine aims to estimate $C_L$ and $C_D$ in real-time to adapt the digital twin to structural damage or environmental changes.

## 3. Numerical Integration (RK4)

To update the state over time, we use the 4th-order Runge-Kutta method:
1. $k_1 = f(t_n, x_n)$
2. $k_2 = f(t_n + \frac{h}{2}, x_n + \frac{h}{2}k_1)$
3. $k_3 = f(t_n + \frac{h}{2}, x_n + \frac{h}{2}k_2)$
4. $k_4 = f(t_n + h, x_n + hk_3)$
5. $x_{n+1} = x_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)$

This provides a stable and accurate simulation of the nonlinear dynamics.
