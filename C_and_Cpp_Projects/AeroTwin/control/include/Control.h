#pragma once
#include <Eigen/Dense>
#include "Dynamics.h"

namespace aerotwin {

class Controller {
public:
    Controller() {
        // Simple PD-like gain matrix for stabilization
        K_.setZero();
        // Pitch control (q) depends on pitch angle and pitch rate
        K_(0, 8) = 5.0;  // Proportional to pitch
        K_(0, 11) = 2.0; // Proportional to pitch rate
    }

    /**
     * @brief Computes control moments based on state error.
     * @param current The current state.
     * @param target The desired target state.
     */
    Eigen::Vector3d computeCommand(const State& current, const State& target);

private:
    Eigen::Matrix<double, 3, 13> K_; // Gain matrix
};

} // namespace aerotwin
