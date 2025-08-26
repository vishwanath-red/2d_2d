cmake_minimum_required(VERSION 3.10)
project(plate12withICP)

# === OpenCV ===
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# === GDCM ===
set(GDCM_DIR "C:/msys64/mingw64/lib/cmake/gdcm")  # <-- Path to GDCMConfig.cmake
find_package(GDCM REQUIRED)

# === Eigen and JSON ===
include_directories(
    "C:/eigen-3.4.0"
    "C:/msys64/mingw64/include/gdcm-3.0"
    "C:/nlohmann"
)

# === Build target ===
add_executable(plate12withICP
    maindcmbatch.cpp
    Center_detection.cpp
    Outlier_angle.cpp
    S_F_B.cpp
    dbscan.cpp
    blob_detection.cpp
    icp_angle.cpp
    Calibration.cpp  # Added Calibration module
    #simulation_4R.cpp
    # spline_fitting.cpp
    # dist_out_removal.cpp
    # tps.cpp
    utils.cpp
    #spline_fitting.cpp
    # plate12withICP_noInputs.cpp
    kmeans.cpp
    plate12icp.cpp
    plate12indexing.cpp
    #plate12icp_fixed.cpp
    #platetest.cpp
    #plate12withICP_exact.cpp
    crop_image.cpp
)

# === Link libraries ===
target_link_libraries(plate12withICP
    ${OpenCV_LIBS}
    gdcmMSFF  
)