#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <chrono>
#include <thread>
#include <string>

#include "Dynamics.h"
#include "Vision.h"
#include "Control.h"
#include "Estimator.h"
#include "NeuralSurrogate.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace aerotwin {
    struct Waypoint {
        Eigen::Vector3d pos; // NED
        std::string label;
    };
}

int main(int argc, char** argv) {
    aerotwin::Parameters params;
    aerotwin::Controller controller;
    aerotwin::Estimator estimator;
    aerotwin::MLSurrogate ml_pinn;

    aerotwin::State state = aerotwin::State::Zero();
    
    // Time tracking
    double total_time = 0.0;
    double dt = 0.01; // 100Hz
    auto step_duration = std::chrono::milliseconds(10);

    while (true) {
        auto start_time = std::chrono::steady_clock::now();
        total_time += dt;

        // --- SCRIPTED FLIGHT ROUTE SELECTION ---
        double radius = 75.0;
        double omega = 0.2;
        double target_alt = 50.0;
        
        std::string path_type = (argc > 1) ? argv[1] : "circle";

        if (total_time < 5.0) {
            // Smooth Takeoff (starting from 0,0,0)
            state.position.z() = -(target_alt * (total_time / 5.0));
            state.position.x() = 0;
            state.position.y() = 0;
            state.velocity.z() = -target_alt / 5.0;
        } else {
            double t_c = total_time - 5.0;
            if (path_type == "climb") {
                state.position.x() = 10.0 * t_c;
                state.position.y() = 0;
                state.position.z() = -target_alt - (2.0 * t_c);
                state.velocity.x() = 10.0;
                state.velocity.z() = -2.0;
            } else if (path_type == "square") {
                double side = 100.0;
                double speed = 15.0;
                double dist = speed * t_c;
                int leg = (int)(dist / side) % 4;
                double pos_in_leg = fmod(dist, side);

                if (leg == 0) { state.position.x() = pos_in_leg; state.position.y() = 0; }
                else if (leg == 1) { state.position.x() = side; state.position.y() = pos_in_leg; }
                else if (leg == 2) { state.position.x() = side - pos_in_leg; state.position.y() = side; }
                else { state.position.x() = 0; state.position.y() = side - pos_in_leg; }
                
                state.position.z() = -target_alt;
                state.velocity.x() = (leg == 0) ? speed : (leg == 2 ? -speed : 0);
                state.velocity.y() = (leg == 1) ? speed : (leg == 3 ? -speed : 0);
            } else {
                // Circle (with smooth transition from origin)
                double ramp = std::min(1.0, t_c / 5.0); // 5s to reach full radius
                state.position.x() = radius * ramp * cos(omega * t_c);
                state.position.y() = radius * ramp * sin(omega * t_c);
                state.position.z() = -target_alt;
                
                state.velocity.x() = -radius * ramp * omega * sin(omega * t_c);
                state.velocity.y() = radius * ramp * omega * cos(omega * t_c);
            }
        }

        // Scripted Orientation: Realistic Coordinated Turn
        double v_horiz = sqrt(state.velocity.x()*state.velocity.x() + state.velocity.y()*state.velocity.y());
        double yaw = (v_horiz > 0.1) ? atan2(state.velocity.y(), state.velocity.x()) : 0.0;
        double target_bank = (v_horiz > 0.1 && path_type == "circle") ? atan2(v_horiz * v_horiz, params.gravity * radius) : 0.0;
        
        state.orientation = Eigen::Quaterniond(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
                                             Eigen::AngleAxisd(0.0, Eigen::Vector3d::UnitY()) *
                                             Eigen::AngleAxisd(target_bank, Eigen::Vector3d::UnitX()));

        // --- INVERSE ENGINE & PREDICTION LOGIC ---
        // Observed "Dummy Force" identifying lift coefficients
        Eigen::Vector3d f_obs(0, 0, -(params.mass * params.gravity)); 
        
        double vx = state.velocity.norm();
        double true_aoa = 5.0 + 2.0 * sin(total_time * 0.5); 
        double vision_aoa = true_aoa + (std::rand() % 100 - 50) / 200.0; 

        estimator.update(state, f_obs, vision_aoa, dt);
        Eigen::Vector3d ml_residuals = ml_pinn.predictResidual(state.toVector());

        // --- JSON TELEMETRY OUTPUT ---
        std::cout << std::fixed << std::setprecision(4)
                  << "{\"t\":" << total_time << ","
                  << "\"px\":" << state.position.x() << ","
                  << "\"py\":" << state.position.y() << ","
                  << "\"pz\":" << state.position.z() << ","
                  << "\"alt\":" << -state.position.z() << ","
                  << "\"vx\":" << vx << ","
                  << "\"qw\":" << state.orientation.w() << ","
                  << "\"qx\":" << state.orientation.x() << ","
                  << "\"qy\":" << state.orientation.y() << ","
                  << "\"qz\":" << state.orientation.z() << ","
                  << "\"cl\":" << estimator.getCL() << ","
                  << "\"cl_target\":" << estimator.getTargetCL() << ","
                  << "\"conf\":" << estimator.getConfidence() << ","
                  << "\"res\":" << ml_residuals.norm() << ","
                  << "\"wp\":\"" << (total_time < 5.0 ? "Takeoff" : "Orbit") << "\"}" << std::endl;

        auto end_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        if (elapsed < step_duration) {
            std::this_thread::sleep_for(step_duration - elapsed);
        }
    }
    return 0;
}
