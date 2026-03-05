#pragma once
#include <Eigen/Dense>
#include <vector>
#include <cmath>

namespace aerotwin {

/**
 * @brief Physics-Informed Neural Network (PINN) Surrogate.
 * Estimates aerodynamic residuals (unmodeled forces) based on state.
 */
class MLSurrogate {
public:
    MLSurrogate() {
        // Mock weights for a small MLP
        weights_ = Eigen::MatrixXd::Random(3, 13) * 0.1;
        bias_ = Eigen::Vector3d::Random() * 0.05;
    }

    /**
     * @brief Inference: Predicts Delta_F based on current state.
     */
    Eigen::Vector3d predictResidual(const Eigen::Matrix<double, 13, 1>& state_vector) {
        // Simple linear layer as a "Neural Operator" placeholder
        // In reality, this would be a loaded ONNX model.
        return weights_ * state_vector + bias_;
    }

private:
    Eigen::MatrixXd weights_;
    Eigen::Vector3d bias_;
};

} // namespace aerotwin
