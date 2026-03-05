#include "Vision.h"
#include <cmath>
#include <random>

namespace aerotwin {

std::vector<FlowFeature> VisionSystem::estimateFlow(const Eigen::Vector3d& body_velocity, const Eigen::Vector3d& angular_velocity) {
    std::vector<FlowFeature> features;
    
    // Create synthetic features representing ground/air flow
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    for (int i = 0; i < 20; ++i) {
        FlowFeature f;
        f.position << distribution(generator), distribution(generator);
        
        // Mock flow field: flow is proportional to velocity and rotation
        // Flow_v = - (V + W x R) / Z  (Simplified)
        f.flow_vector << -body_velocity.x() * 0.05 + angular_velocity.y() * 0.1,
                         -body_velocity.y() * 0.05 - angular_velocity.x() * 0.1;
        
        features.push_back(f);
    }

    return features;
}

double VisionSystem::estimateAoA(const std::vector<FlowFeature>& features) {
    if (features.empty()) return 0.0;

    // Heuristic: AoA is related to the ratio of vertical to horizontal flow components
    double sum_vx = 0, sum_vz = 0;
    for (const auto& f : features) {
        sum_vx += std::abs(f.flow_vector.x());
        sum_vz += f.flow_vector.y();
    }
    
    // Very simplified mapping for demonstration
    return std::atan2(sum_vz, sum_vx) * 180.0 / 3.14159;
}

} // namespace aerotwin
