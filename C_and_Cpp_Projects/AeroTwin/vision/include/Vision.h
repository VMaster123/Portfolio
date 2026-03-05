#pragma once
#include <Eigen/Dense>
#include <vector>

namespace aerotwin {

/**
 * @brief Representation of a visual feature in the flow field.
 */
struct FlowFeature {
    Eigen::Vector2d position;     // Normalized image coordinates
    Eigen::Vector2d flow_vector;  // Optical flow velocity
};

class VisionSystem {
public:
    VisionSystem() {}

    /**
     * @brief Simulates optical flow estimation from drone motion.
     * In a full implementation, this would use OpenCV to process camera frames.
     */
    std::vector<FlowFeature> estimateFlow(const Eigen::Vector3d& body_velocity, const Eigen::Vector3d& angular_velocity);

    /**
     * @brief Estimates Angle of Attack (AoA) from visual flow features.
     */
    double estimateAoA(const std::vector<FlowFeature>& features);
};

} // namespace aerotwin
