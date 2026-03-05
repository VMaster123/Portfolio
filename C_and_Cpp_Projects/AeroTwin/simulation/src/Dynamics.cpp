#include "Dynamics.h"

namespace aerotwin {

Eigen::Matrix<double, 13, 1> Dynamics::computeDerivative(const State& state, const Eigen::Vector3d& forces, const Eigen::Vector3d& moments) {
    Eigen::Matrix<double, 13, 1> dx;

    // 1. Translational Position Derivative: dr = R(q) * v
    dx.segment<3>(0) = state.orientation.toRotationMatrix() * state.velocity;

    // 2. Body Velocity Derivative: dv = F/m + R^T*g - w x v
    // Transform gravity from NED inertial frame to body frame
    Eigen::Vector3d g_body = state.orientation.inverse() * Eigen::Vector3d(0, 0, params_.gravity);
    dx.segment<3>(3) = (forces / params_.mass) + g_body - state.angular_velocity.cross(state.velocity);

    // 3. Quaternion Derivative: dq = 0.5 * q * [0, w]
    Eigen::Quaterniond w_quat(0, state.angular_velocity.x(), state.angular_velocity.y(), state.angular_velocity.z());
    Eigen::Quaterniond dq = state.orientation * w_quat;
    dx(6) = 0.5 * dq.w();
    dx(7) = 0.5 * dq.x();
    dx(8) = 0.5 * dq.y();
    dx(9) = 0.5 * dq.z();

    // 4. Angular Velocity Derivative: dw = I^-1 * (M - w x (Iw))
    Eigen::Vector3d angular_momentum = params_.inertia * state.angular_velocity;
    dx.segment<3>(10) = params_.inertia.inverse() * (moments - state.angular_velocity.cross(angular_momentum));

    return dx;
}

State Dynamics::step(const State& state, const Eigen::Vector3d& forces, const Eigen::Vector3d& moments, double dt) {
    auto f = [&](const State& s) {
        return computeDerivative(s, forces, moments);
    };

    Eigen::Matrix<double, 13, 1> k1 = f(state);
    Eigen::Matrix<double, 13, 1> k2 = f(State::fromVector(state.toVector() + 0.5 * dt * k1));
    Eigen::Matrix<double, 13, 1> k3 = f(State::fromVector(state.toVector() + 0.5 * dt * k2));
    Eigen::Matrix<double, 13, 1> k4 = f(State::fromVector(state.toVector() + dt * k3));

    Eigen::Matrix<double, 13, 1> next_v = state.toVector() + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    
    State next_state = State::fromVector(next_v);
    next_state.orientation.normalize(); // Ensure unit quaternion
    return next_state;
}

} // namespace aerotwin
