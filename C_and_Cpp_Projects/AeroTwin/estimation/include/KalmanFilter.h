#ifndef KALMAN_FILTER_H
#define KALMAN_FILTER_H

#include <Eigen/Dense>
#include "Dynamics.h"

namespace aerotwin {

    /**
     * @brief Extended Kalman Filter (EKF) for UAV State Estimation.
     * Estimates Position, Velocity, and Orientation from noisy sensors.
     */
    class StateEstimator {
    public:
        StateEstimator();
        
        // Predict the next state based on IMU inputs (Acceleration and Angular Rate)
        void predict(const Eigen::Vector3d& acc, const Eigen::Vector3d& gyro, double dt);
        
        // Correct the state based on GPS/Baro measurements (Position)
        void updateGPS(const Eigen::Vector3d& pos_measured);
        
        // Correct the orientation based on Magnetometer or Vision Data
        void updateOrientation(const Eigen::Quaterniond& q_measured);

        // Get the current estimated state
        State getState() const;
        
        // Get Estimation Confidence (Trace of Covariance)
        double getConfidence() const;

    private:
        Eigen::VectorXd x_hat; // State vector: [px, py, pz, vx, vy, vz, qw, qx, qy, qz]
        Eigen::MatrixXd P;     // State Covariance
        Eigen::MatrixXd Q;     // Process Noise Covariance
        Eigen::MatrixXd R_gps; // Measurement Noise for GPS
        
        void normalizeQuaternion();
    };

}

#endif
