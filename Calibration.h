#ifndef CALIBRATION_H
#define CALIBRATION_H

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <filesystem>

struct CalibrationResult {
    Eigen::Matrix4d Result_Matrix;
    double RPE;
    bool success;
    std::string error_message;
};

struct TransformationData {
    double tx, ty, tz;
    std::vector<double> rotation; // quaternion [x, y, z, w]
};

class Calibration {
public:
    // Main calibration function
    static CalibrationResult calibrate(
        const std::string& position,
        const TransformationData& C2R,
        const TransformationData& C2D,
        const Eigen::MatrixXd& W,
        const Eigen::MatrixXd& Dpts,
        double r,
        const std::string& path_d,
        int rewarp = 0
    );

private:
    // Helper functions
    static std::pair<Eigen::MatrixXd, Eigen::MatrixXd> refConversion(
        const std::string& position,
        const Eigen::MatrixXd& W,
        const Eigen::MatrixXd& distpts,
        const TransformationData& C2R,
        const TransformationData& C2D,
        const std::string& path_d
    );

    static std::pair<Eigen::Matrix<double, 3, 4>, std::vector<double>> estimateCameraMatrix(
        const Eigen::MatrixXd& imagePoints,
        const Eigen::MatrixXd& worldPoints
    );

    static std::tuple<Eigen::Matrix3d, Eigen::Matrix3d, Eigen::Vector3d, Eigen::Vector2d, Eigen::Vector3d> 
    decomposeCamera(const Eigen::Matrix<double, 3, 4>& P);

    static double registrationCheck(
        const Eigen::Matrix<double, 3, 4>& P,
        const Eigen::MatrixXd& W_dist_pts,
        const std::string& position,
        double r,
        const std::string& path_d,
        int rewarp
    );

    // Utility functions
    static Eigen::Matrix3d quaternionToRotationMatrix(const std::vector<double>& q);
    static std::pair<Eigen::MatrixXd, Eigen::Matrix3d> normalise2dpts(const Eigen::MatrixXd& pts);
    static std::pair<Eigen::Matrix3d, Eigen::Matrix3d> rq3(const Eigen::Matrix3d& A);
    static Eigen::MatrixXd readMatrix(const std::string& filename);
    static void writeMatrix(const std::string& filename, const Eigen::MatrixXd& matrix, 
                           const std::string& delimiter = " ", int precision = 6);
    static void appendMatrix(const std::string& filename, const Eigen::MatrixXd& matrix, 
                            const std::string& delimiter = " ", int precision = 6);
    static bool fileExists(const std::string& filename);
    static void writeErrorLog(const std::string& message, const std::string& path_d, 
                             const std::string& position);
    static void appLog(const std::string& operation, const std::string& path_d);
    
    // Load ICP fiducial data (equivalent to load icpPlatFid.mat)
    static Eigen::MatrixXd loadIcpPlatFid();
};

#endif // CALIBRATION_H