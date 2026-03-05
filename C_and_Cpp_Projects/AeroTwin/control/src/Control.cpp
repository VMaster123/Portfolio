#include "Control.h"

namespace aerotwin {

Eigen::Vector3d Controller::computeCommand(const State& current, const State& target) {
    // x_error = target - current
    Eigen::Matrix<double, 13, 1> error = target.toVector() - current.toVector();
    
    // Moments = K * error
    // We only care about rotation gains for this simplified model
    Eigen::Vector3d moments;
    moments.setZero();
    
    // Pitch control
    moments.y() = 2.0 * error(8) + 0.5 * error(11); // Simplified PD
    
    // Roll control
    moments.x() = 2.0 * error(7) + 0.5 * error(10);
    
    // Yaw control
    moments.z() = 1.0 * error(9) + 0.5 * error(12);

    return moments;
}

} // namespace aerotwin
