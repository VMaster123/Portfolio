cmake_minimum_required(VERSION 3.10)
project(AeroTwin)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen3
find_package(Eigen3 REQUIRED CONFIG)

# Find Threads
find_package(Threads REQUIRED)

# ==========================================
# Core Library: AeroTwin Modules
# ==========================================
add_library(aerotwin_core 
    simulation/src/Dynamics.cpp
    vision/src/Vision.cpp
    control/src/Control.cpp
    inverse/src/Estimator.cpp
)

target_include_directories(aerotwin_core PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/simulation/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/vision/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/control/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inverse/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ml/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/backend/include>
)

target_link_libraries(aerotwin_core PUBLIC 
    Eigen3::Eigen
)

# ==========================================
# Executable: AeroTwin Simulator
# ==========================================
add_executable(aerotwin_sim 
    simulation/main.cpp
)

target_link_libraries(aerotwin_sim PRIVATE 
    aerotwin_core
)

# ==========================================
# Executable: Avionics Logic
# ==========================================
add_executable(avionics 
    avionics.cpp
)

target_link_libraries(avionics PRIVATE 
    Threads::Threads
)