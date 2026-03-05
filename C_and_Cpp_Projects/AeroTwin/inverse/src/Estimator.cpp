#include "Estimator.h"
#include <iostream>

namespace aerotwin {

void Estimator::update(const State& state, const Eigen::Vector3d& total_force_applied, double aoa, double dt) {
    double airspeed = state.velocity.norm();
    if (airspeed < 2.0) {
        confidence_ *= 0.95; // Decay confidence while stationary
        return; 
    }

    // 1. DYNAMIC SYSTEM INVERSION
    // We observe the "Total Force" applied to the body.
    // F_total = F_thrust + F_aero + F_residual
    // Lift is the vertical component of the aerodynamic force (in Body NED, z is down, lift is -z)
    // Here we use the "Reference Model" to identify the target CL from flow estimation
    double alpha_rad = aoa * 3.14159 / 180.0;
    target_cl_ref_ = (std::abs(total_force_applied.z()) / (0.5 * 1.225 * airspeed * airspeed * 0.1));
    
    // Bounds for stability
    if (target_cl_ref_ > 2.0) target_cl_ref_ = 2.0;

    // 2. RECURSIVE UPDATER (RLS-Lite)
    double innovation = target_cl_ref_ - CL_est_;
    CL_est_ += 0.8 * innovation * dt;
    
    // 3. CONFIDENCE MONITORING
    // Accuracy = 1 - Normalized Error
    double error_score = std::abs(innovation) / (target_cl_ref_ + 0.1);
    double instant_conf = std::max(0.0, std::min(1.0, 1.0 - error_score));
    
    // Temporal Smoothing (Confidence shouldn't jump to 100% instantly)
    confidence_ += 0.2 * (instant_conf - confidence_) * dt;
    
    // Add jitter if airspeed is high (simulating turbulence/sensor noise)
    if (airspeed > 15.0) {
        confidence_ *= (0.98 + (std::rand() % 4 / 100.0));
    }
    
    // Drag update Cd = f(airspeed, acceleration_x)
    double target_cd = 0.05 + 0.1 * CL_est_ * CL_est_;
    CD_est_ += 0.5 * (target_cd - CD_est_) * dt;
}

} // namespace aerotwin
