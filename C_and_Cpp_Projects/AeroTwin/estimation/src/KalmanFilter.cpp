#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include "KalmanFilter.h"

namespace aerotwin {

    StateEstimator::StateEstimator() {
        // Initial State: Position (0,0,0), Velocity (0,0,0), Quaternion (Identity)
        x_hat = Eigen::VectorXd::Zero(10);
        x_hat(6) = 1.0; // qw = 1 (Identity quaternion)
        
        // Initial Covariance: High uncertainty initially
        P = Eigen::MatrixXd::Identity(10, 10) * 1.0;
        
        // Process Noise (Q): How much we trust our model dynamics
        Q = Eigen::MatrixXd::Identity(10, 10) * 0.01;
        
        // Measurement Noise (R): How noisy the GPS is (standard for low-cost GPS)
        R_gps = Eigen::MatrixXd::Identity(3, 3) * 0.1; 
    }

    void StateEstimator::predict(const Eigen::Vector3d& acc, const Eigen::Vector3d& gyro, double dt) {
        // 1. Prediction Step: Integration of Velocity and Acceleration
        // Update Position: p = p + v*dt + 0.5*a*dt^2
        x_hat(0) += x_hat(3) * dt + 0.5 * acc(0) * dt * dt;
        x_hat(1) += x_hat(4) * dt + 0.5 * acc(1) * dt * dt;
        x_hat(2) += x_hat(5) * dt + 0.5 * acc(2) * dt * dt;
        
        // Update Velocity: v = v + a*dt
        x_hat(3) += acc(0) * dt;
        x_hat(4) += acc(1) * dt;
        x_hat(5) += acc(2) * dt;

        // Quaternion Propogation based on Angular Rates (gyro)
        // Correct way is to multiply current q by rotation q(gyro*dt)
        Eigen::Quaterniond q_curr(x_hat(6), x_hat(7), x_hat(8), x_hat(9));
        Eigen::Vector3d rot_vec = gyro * dt * 0.5;
        Eigen::Quaterniond q_rot(1.0, rot_vec.x(), rot_vec.y(), rot_vec.z());
        Eigen::Quaterniond q_next = (q_curr * q_rot).normalized();
        
        x_hat(6) = q_next.w();
        x_hat(7) = q_next.x();
        x_hat(8) = q_next.y();
        x_hat(9) = q_next.z();

        // 2. Covariance Update: P_next = F*P*F' + Q
        // (Simplified: Constant F for this demonstration)
        P = P + Q; 
    }

    void StateEstimator::updateGPS(const Eigen::Vector3d& pos_measured) {
        // Measurement Matrix (H): Mapping 10D state to 3D position
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, 10);
        H.block<3, 3>(0, 0) = Eigen::MatrixXd::Identity(3, 3);
        
        // Innovation: z - H*x_hat
        Eigen::Vector3d z = pos_measured;
        Eigen::Vector3d y = z - H * x_hat;
        
        // Innovation Covariance (S) and Kalman Gain (K)
        Eigen::MatrixXd S = H * P * H.transpose() + R_gps;
        Eigen::MatrixXd K = P * H.transpose() * S.inverse();
        
        // Correct State and Covariance
        x_hat = x_hat + K * y;
        P = (Eigen::MatrixXd::Identity(10, 10) - K * H) * P;
        
        normalizeQuaternion();
    }

    void StateEstimator::updateOrientation(const Eigen::Quaterniond& q_measured) {
        // Measurement Matrix (H) for Quaternions: mapping bottom 4 states
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(4, 10);
        H.block<4, 4>(0, 6) = Eigen::MatrixXd::Identity(4, 4);
        
        Eigen::Vector4d z(q_measured.w(), q_measured.x(), q_measured.y(), q_measured.z());
        Eigen::Vector4d y = z - H * x_hat;
        
        Eigen::MatrixXd R_q = Eigen::MatrixXd::Identity(4, 4) * 0.05;
        Eigen::MatrixXd S = H * P * H.transpose() + R_q;
        Eigen::MatrixXd K = P * H.transpose() * S.inverse();
        
        x_hat = x_hat + K * y;
        P = (Eigen::MatrixXd::Identity(10, 10) - K * H) * P;

        normalizeQuaternion();
    }

    State StateEstimator::getState() const {
        aerotwin::State s;
        s.position = x_hat.segment<3>(0);
        s.velocity = x_hat.segment<3>(3);
        s.orientation = Eigen::Quaterniond(x_hat(6), x_hat(7), x_hat(8), x_hat(9));
        return s;
    }

    double StateEstimator::getConfidence() const {
        return 1.0 / (1.0 + P.diagonal().sum()); // Higher sum = lower confidence
    }

    void StateEstimator::normalizeQuaternion() {
        double mag = sqrt(x_hat(6)*x_hat(6) + x_hat(7)*x_hat(7) + x_hat(8)*x_hat(8) + x_hat(9)*x_hat(9));
        if (mag > 0) {
            x_hat.segment<4>(6) /= mag;
        }
    }
}
