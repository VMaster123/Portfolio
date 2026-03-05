#pragma once
#include <Eigen/Dense>
#include "Dynamics.h"

namespace aerotwin {

/**
 * @brief Estimator for aerodynamic coefficients via Recursive Least Squares or EKF.
 */
class Estimator {
public:
    Estimator() : CL_est_(0.5), CD_est_(0.1) {}

    /**
     * @brief Updates estimates based on observed acceleration and known thrust.
     */
    void update(const State& state, const Eigen::Vector3d& total_force, double aoa, double dt);

    double getCL() const { return CL_est_; }
    double getCD() const { return CD_est_; }
    double getTargetCL() const { return target_cl_ref_; }
    double getConfidence() const { return confidence_; }

private:
    double CL_est_;
    double CD_est_;
    double target_cl_ref_ = 0.5;
    double confidence_ = 0.0;
};

} // namespace aerotwin
