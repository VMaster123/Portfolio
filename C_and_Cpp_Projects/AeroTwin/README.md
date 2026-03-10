# AeroTwin (Web Edition)

### High-Performance Digital Twin for UAV Aerodynamics
**Remade to be 100% dependency-free (No C++, No vcpkg, No CMake).**

AeroTwin-NX is a web-native simulation and engineering platform. It implements a **6-DOF flight dynamics engine** directly in Node.js, streaming real-time telemetry to a **premium GLSL-accelerated 3D dashboard**.

------------------------------------------------------------------------

## 🚀 Getting Started (No Build Required)

One command to rule them all. No C++ compiler needed.

1.  **Install dependencies**:
    ```bash
    npm install
    ```
2.  **Launch the System**:
    ```bash
    npm start
    ```
3.  **Open the Dashboard**:
    Go to [http://localhost:8080](http://localhost:8080)

------------------------------------------------------------------------

## 🏗 New Architecture (vcpkg-Free)

-   **Physics Engine (`server/sim.js`)**: Ported from C++/Eigen. Handles 6-DOF Rigid Body Dynamics, RK4 integration, and Coordinated Turn modeling.
-   **Backend (`server/index.js`)**: Lightweight Node.js server. Broadcasts telemetry via high-frequency WebSockets.
-   **Dashboard (`frontend/index.html`)**: A "Premium Aesthetics" visualizer using **Three.js** (WebGL), **Chart.js**, and **Glassmorphism** styling.

------------------------------------------------------------------------

## 🧪 Simulation Modes

Control the flight route by passing URL parameters:
-   **Circle Orbit**: `http://localhost:8080/?path=circle`
-   **Square Pattern**: `http://localhost:8080/?path=square`
-   **Steady Climb**: `http://localhost:8080/?path=climb`

------------------------------------------------------------------------

## 🧠 Technical Domains

-   **State Estimation**: Simulated EKF (Extended Kalman Filter) confidence metrics.
-   **Control Theory**: Coordinated bank-to-turn flight logic.
-   **Computer Vision (Synthetic)**: Visual flow field vectorization (Simulated).
-   **Web Graphics**: High-fidelity Three.js rendering with trajectory trails and real-time Chart.js integration.

------------------------------------------------------------------------

## 📜 License
MIT License

## 👤 Author
Developed as a modern web-native remaking of the AeroTwin robotics project.

