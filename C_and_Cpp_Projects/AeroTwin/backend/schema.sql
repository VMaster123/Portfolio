-- AeroTwin Time-Series Database Schema
-- Optimized for TimescaleDB (PostgreSQL Extension)

-- 1. Enable TimescaleDB Extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- 2. Flight Telemetry Table
CREATE TABLE flight_telemetry (
    time        TIMESTAMPTZ       NOT NULL,
    uav_id      UUID              NOT NULL,
    
    -- State: Position and Orientation
    pos_x       DOUBLE PRECISION  NOT NULL,
    pos_y       DOUBLE PRECISION  NOT NULL,
    pos_z       DOUBLE PRECISION  NOT NULL,
    quat_w      DOUBLE PRECISION,
    quat_x      DOUBLE PRECISION,
    quat_y      DOUBLE PRECISION,
    quat_z      DOUBLE PRECISION,
    
    -- Aerodynamic Estimates
    cl_est      DOUBLE PRECISION,
    cd_est      DOUBLE PRECISION,
    ml_res_z    DOUBLE PRECISION,
    
    -- Performance Tracking
    latency_ms  INTEGER
);

-- 3. Convert to Hypertable (Partitioned by time)
SELECT create_hypertable('flight_telemetry', 'time');

-- 4. Continuous Aggregate: 1-Second Averages for Dashboard
CREATE MATERIALIZED VIEW telemetry_stats_1s
WITH (timescaledb.continuous = true) AS
SELECT time_bucket('1 second', time),
       uav_id,
       avg(abs(pos_z)) as avg_altitude,
       avg(cl_est) as avg_cl,
       max(latency_ms) as max_latency
FROM flight_telemetry
GROUP BY 1, 2;

-- 5. Index for spatial/UAV-based queries
CREATE INDEX idx_uav_time ON flight_telemetry (uav_id, time DESC);
