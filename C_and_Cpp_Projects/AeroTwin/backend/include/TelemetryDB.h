#pragma once
#include <string>
#include <vector>
#include <iostream>

namespace aerotwin {

/**
 * @brief Telemetry Schema for Time-Series Storage (TimescaleDB style).
 */
struct TelemetryPacket {
    double timestamp;
    double px, py, pz;
    double vx, vy, vz;
    double qw, qx, qy, qz;
    double cl_est, cd_est;
};

class TelemetryDB {
public:
    /**
     * @brief Mimics an INSERT into a PostgreSQL/TimescaleDB hypertable.
     */
    void log(const TelemetryPacket& p) {
        // SQL Concept: INSERT INTO telemetry (time, pos_x, ...) VALUES (...)
        // time_bucket('1 second', time) for continuous aggregates
        buffer_.push_back(p);
    }

    void dumpSchema() {
        std::cout << "[DATABASE SCHEMA] CREATE TABLE flight_telemetry (" << std::endl;
        std::cout << "  time TIMESTAMPTZ NOT NULL," << std::endl;
        std::cout << "  uav_id UUID," << std::endl;
        std::cout << "  position GEOMETRY(POINTZ, 4326)," << std::endl;
        std::cout << "  cl_estimate DOUBLE PRECISION," << std::endl;
        std::cout << "  cd_estimate DOUBLE PRECISION" << std::endl;
        std::cout << ");" << std::endl;
        std::cout << "[DATABASE SCHEMA] SELECT create_hypertable('flight_telemetry', 'time');" << std::endl;
    }

private:
    std::vector<TelemetryPacket> buffer_;
};

} // namespace aerotwin
