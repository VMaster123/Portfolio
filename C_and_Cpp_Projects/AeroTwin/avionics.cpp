#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>
#include <vector>
#include <mutex>

// =====================
// Mock sensor definitions
// =====================
struct SensorData {
    double altitude;    // meters
    double airspeed;    // m/s
    double pitch;       // degrees
    double roll;        // degrees
    double yaw;         // degrees
};

// =====================
// Shared system state
// =====================
std::atomic<bool> systemRunning(true);
std::mutex sensorMutex;
SensorData currentSensorData;

// =====================
// Sensor simulation thread
// =====================
void readSensors() {
    while (systemRunning) {
        std::lock_guard<std::mutex> lock(sensorMutex);
        // Simulate sensor readings (replace with real sensors)
        currentSensorData.altitude += 0.1;
        currentSensorData.airspeed += 0.05;
        currentSensorData.pitch = 1.0;
        currentSensorData.roll = 0.5;
        currentSensorData.yaw = 0.0;

        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 10 Hz
    }
}

// =====================
// Control loop thread
// =====================
void controlLoop() {
    while (systemRunning) {
        SensorData localCopy;
        {
            std::lock_guard<std::mutex> lock(sensorMutex);
            localCopy = currentSensorData;
        }

        // Simple control logic placeholder
        double pitchCommand = -0.1 * localCopy.pitch;
        double rollCommand = -0.1 * localCopy.roll;

        // Normally send commands to actuators here
        std::cout << "Pitch Command: " << pitchCommand
                  << ", Roll Command: " << rollCommand
                  << ", Altitude: " << localCopy.altitude
                  << ", Airspeed: " << localCopy.airspeed
                  << std::endl;

        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 10 Hz
    }
}

// =====================
// Main avionics system
// =====================
int main() {
    std::cout << "Starting Avionics Software..." << std::endl;

    // Launch threads
    std::thread sensorThread(readSensors);
    std::thread controlThread(controlLoop);

    // Run system for 10 seconds
    std::this_thread::sleep_for(std::chrono::seconds(10));
    systemRunning = false;

    sensorThread.join();
    controlThread.join();

    std::cout << "Avionics Software Stopped." << std::endl;
    return 0;
}
