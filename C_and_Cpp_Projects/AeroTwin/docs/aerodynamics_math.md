# AeroTwin: Advanced Mathematical Foundations

## 1. Computer Vision: Flow Feature Extraction
The relationship between visual flow and aircraft motion is modeled through the **Interaction Matrix** $\mathbf{L}_s$:

$$\dot{\mathbf{s}} = \mathbf{L}_s(\mathbf{s}, Z) \mathbf{v}_c$$

Where:
- $\mathbf{s} = [x, y]^T$ is the image coordinate.
- $Z$ is the depth.
- $\mathbf{v}_c = [v_x, v_y, v_z, \omega_x, \omega_y, \omega_z]^T$ is the camera velocity.

The interaction matrix is defined as:
$$\mathbf{L}_s = \begin{bmatrix} -1/Z & 0 & x/Z & xy & -(1+x^2) & y \\ 0 & -1/Z & y/Z & 1+y^2 & -xy & -x \end{bmatrix}$$

AeroTwin uses this to invert the problem: estimating $v_c$ from observed $\dot{\mathbf{s}}$ to validate onboard IMU data.

## 2. ML: PINN Gradient Constraints
The Neural Surrogate is trained by minimizing a composite loss function that enforces physical laws:

$$\mathcal{L} = \mathcal{L}_{data} + \lambda_{physics} \mathcal{L}_{physics}$$

The physics loss $\mathcal{L}_{physics}$ enforces the Navier-Stokes residual or Newton-Euler dynamic consistency:
$$\mathcal{L}_{physics} = \left\| \mathbf{I}\dot{\omega} + \omega \times (\mathbf{I}\omega) - (\mathbf{M}_{aero} + \mathbf{M}_{pinn}) \right\|^2$$

By utilizing **Automatic Differentiation**, the PINN learns the mapping from state to residual force while remaining bounded by the conservation of angular momentum.

## 3. Database: Continuous Aggregates for Telemetry
To handle 1kHz telemetry streams, we utilize **TimescaleDB Continuous Aggregates**. This reduces data volume for the frontend by pre-computing 100ms buckets of aerodynamic coefficients:

```sql
CREATE MATERIALIZED VIEW aero_trends_100ms
WITH (timescaledb.continuous = true) AS
SELECT time_bucket('100 milliseconds', time),
       avg(cl_estimate) as cl_avg,
       stddev(cl_estimate) as cl_std
FROM flight_telemetry
GROUP BY 1;
```
