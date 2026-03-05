#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>

namespace aerotwin {

/**
 * @brief 13-element state vector for 6-DOF rigid body.
 * [px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz]
 */
struct State {
    Eigen::Vector3d position;        // NED frame
    Eigen::Vector3d velocity;        // Body frame
    Eigen::Quaterniond orientation; // B to I (body to inertial)
    Eigen::Vector3d angular_velocity; // Body frame

    static State Zero() {
        State s;
        s.position.setZero();
        s.velocity.setZero();
        s.orientation.setIdentity();
        s.angular_velocity.setZero();
        return s;
    }

    Eigen::Matrix<double, 13, 1> toVector() const {
        Eigen::Matrix<double, 13, 1> v;
        v.segment<3>(0) = position;
        v.segment<3>(3) = velocity;
        v(6) = orientation.w();
        v(7) = orientation.x();
        v(8) = orientation.y();
        v(9) = orientation.z();
        v.segment<3>(10) = angular_velocity;
        return v;
    }

    static State fromVector(const Eigen::Matrix<double, 13, 1>& v) {
        State s;
        s.position = v.segment<3>(0);
        s.velocity = v.segment<3>(3);
        s.orientation = Eigen::Quaterniond(v(6), v(7), v(8), v(9)).normalized();
        s.angular_velocity = v.segment<3>(10);
        return s;
    }
};

struct Parameters {
    double mass = 1.5; // kg
    Eigen::Matrix3d inertia = Eigen::Vector3d(0.015, 0.015, 0.03).asDiagonal();
    double gravity = 9.81;
    double rho = 1.225; // air density kg/m^3
    double S = 0.1; // surface area
};

class Dynamics {
public:
    Dynamics(const Parameters& params) : params_(params) {}

    /**
     * @brief Compute the state derivative f(x, u)
     */
    Eigen::Matrix<double, 13, 1> computeDerivative(const State& state, const Eigen::Vector3d& forces, const Eigen::Vector3d& moments);

    /**
     * @brief RK4 integration step
     */
    State step(const State& state, const Eigen::Vector3d& forces, const Eigen::Vector3d& moments, double dt);

private:
    Parameters params_;
};

} // namespace aerotwin
