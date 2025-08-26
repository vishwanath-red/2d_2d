#include "Calibration.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

CalibrationResult Calibration::calibrate(
    const std::string& position,
    const TransformationData& C2R,
    const TransformationData& C2D,
    const Eigen::MatrixXd& W,
    const Eigen::MatrixXd& Dpts,
    double r,
    const std::string& path_d,
    int rewarp) {
    
    CalibrationResult result;
    result.success = false;
    result.RPE = 10.0;
    
    try {
        // Determine folder based on rewarp flag
        std::string folder = (rewarp == 1) ? "Rewarp\\" : "";
        
        // Load ICP fiducial data (equivalent to load icpPlatFid.mat)
        Eigen::MatrixXd icpPlatFid = loadIcpPlatFid();
        
        // Combine W with icpPlatFid if needed
        // Eigen::MatrixXd W_combined(W.rows() + icpPlatFid.rows(), W.cols());
        // W_combined << W, icpPlatFid;
        
        // Reference conversion
        auto [Ref_CMM_pts, Ref_dist_pts] = refConversion(position, W, Dpts, C2R, C2D, path_d);
        
        // Read 2D points from file
        std::string xy_file = path_d + "\\Output\\" + position + "\\PD\\" + position + "_2D.txt";
        Eigen::MatrixXd xy_o = readMatrix(xy_file);
        
        if (xy_o.rows() == 0) {
            throw std::runtime_error("Failed to read 2D points file: " + xy_file);
        }
        
        // Sort rows in descending order based on the third column
        std::vector<std::pair<double, int>> sortData;
        for (int i = 0; i < xy_o.rows(); ++i) {
            sortData.push_back({xy_o(i, 2), i});
        }
        std::sort(sortData.begin(), sortData.end(), std::greater<std::pair<double, int>>());
        
        Eigen::MatrixXd xy_sorted(xy_o.rows(), xy_o.cols());
        Eigen::VectorXd order(xy_o.rows());
        for (int i = 0; i < xy_o.rows(); ++i) {
            int originalIndex = sortData[i].second;
            xy_sorted.row(i) = xy_o.row(originalIndex);
            order(i) = xy_o(originalIndex, 2);
        }
        
        // Extract xy coordinates and apply coordinate transformation
        Eigen::MatrixXd xy(xy_sorted.rows(), 2);
        xy.col(0) = xy_sorted.col(0);
        xy.col(1) = r - xy_sorted.col(1).array(); // Flip y-coordinate
        
        // Reorder Ref_CMM_pts according to order
        Eigen::MatrixXd Ref_CMM_pts_ordered(order.size(), Ref_CMM_pts.cols());
        for (int i = 0; i < order.size(); ++i) {
            int idx = static_cast<int>(order(i)) - 1; // Convert to 0-based indexing
            if (idx >= 0 && idx < Ref_CMM_pts.rows()) {
                Ref_CMM_pts_ordered.row(i) = Ref_CMM_pts.row(idx);
            }
        }
        
        // Estimate camera matrix
        auto [P0, reprojection_errors] = estimateCameraMatrix(xy, Ref_CMM_pts_ordered);
        
        // Decompose camera matrix
        auto [K, R_ct, Pc1, pp1, pv1] = decomposeCamera(P0);
        
        // Normalize K matrix
        Eigen::Matrix3d K_norm = K / K(2, 2);
        
        // Create Rt3 matrix
        Eigen::Matrix<double, 3, 4> Rt3;
        Rt3.block<3, 3>(0, 0) = R_ct;
        Rt3.block<3, 1>(0, 3) = -R_ct * Pc1;
        
        // Calculate normalized projection matrix
        Eigen::Matrix<double, 3, 4> P_norm = K_norm * Rt3;
        
        // Reshape to 1x12 vector (row-major order)
        Eigen::VectorXd P_patient(12);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 4; ++j) {
                P_patient(i * 4 + j) = P_norm(i, j);
            }
        }
        
        // Write P_patient to file
        std::string p_file = path_d + "\\Output\\" + position + folder + "\\P_Imf" + position + ".txt";
        writeMatrix(p_file, P_patient.transpose(), " ", 6);
        
        // Handle P_Imf.txt file writing based on position
        std::string main_p_file = path_d + "\\Output" + folder + "\\P_Imf.txt";
        
        if (position == "AP") {
            if (fileExists(main_p_file)) {
                Eigen::MatrixXd P2 = readMatrix(main_p_file);
                if (P2.rows() < 2) {
                    appendMatrix(main_p_file, P_patient.transpose(), " ", 6);
                }
            } else {
                writeMatrix(main_p_file, P_patient.transpose(), " ", 6);
            }
        } else if (position == "LP") {
            if (fileExists(main_p_file)) {
                Eigen::MatrixXd P2 = readMatrix(main_p_file);
                if (P2.rows() < 2) {
                    appendMatrix(main_p_file, P_patient.transpose(), " ", 6);
                }
            } else {
                writeMatrix(main_p_file, P_patient.transpose(), " ", 6);
            }
        }
        
        // Registration check
        double error = registrationCheck(P0, Ref_dist_pts, position, r, path_d, rewarp);
        
        // Prepare result
        result.Result_Matrix = Eigen::Matrix4d::Identity();
        result.Result_Matrix.block<3, 4>(0, 0) = P0;
        result.RPE = std::round(error * 1000.0) / 1000.0; // Round to 3 decimal places
        result.success = true;
        
        // Log success
        appLog("CALIBRATION", path_d);
        
    } catch (const std::exception& e) {
        result.success = false;
        result.RPE = 10.0;
        result.error_message = e.what();
        
        // Write error logs
        writeErrorLog("Calibration Failure: " + std::string(e.what()), path_d, position);
        
        std::cerr << "Calibration error: " << e.what() << std::endl;
    }
    
    return result;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> Calibration::refConversion(
    const std::string& position,
    const Eigen::MatrixXd& W,
    const Eigen::MatrixXd& distpts,
    const TransformationData& C2R,
    const TransformationData& C2D,
    const std::string& path_d) {
    
    // Reference marker transformation
    Eigen::Vector3d trans(C2R.tx, C2R.ty, C2R.tz);
    Eigen::Matrix3d rot = quaternionToRotationMatrix({-C2R.rotation[3], C2R.rotation[0], C2R.rotation[1], C2R.rotation[2]});
    
    Eigen::Matrix4d cam2ref = Eigen::Matrix4d::Identity();
    cam2ref.block<3, 3>(0, 0) = rot;
    cam2ref.block<3, 1>(0, 3) = trans;
    
    // Detector marker transformation
    Eigen::Vector3d trans2(C2D.tx, C2D.ty, C2D.tz);
    Eigen::Matrix3d rot2 = quaternionToRotationMatrix({-C2D.rotation[3], C2D.rotation[0], C2D.rotation[1], C2D.rotation[2]});
    
    Eigen::Matrix4d cam2DD = Eigen::Matrix4d::Identity();
    cam2DD.block<3, 3>(0, 0) = rot2;
    cam2DD.block<3, 1>(0, 3) = trans2;
    
    // Calculate transformations
    Eigen::Matrix4d ref2cam = cam2ref.inverse();
    Eigen::Matrix4d ref2DD = ref2cam * cam2DD;
    
    // Save ref2dd.txt for AP position
    if (position == "AP") {
        std::string ref2dd_file = path_d + "\\Output\\" + position + "\\ref2dd.txt";
        writeMatrix(ref2dd_file, ref2DD, " ", 6);
    }
    
    // Transform world points
    Eigen::MatrixXd W_homogeneous(4, W.rows());
    W_homogeneous.block(0, 0, 3, W.rows()) = W.transpose();
    W_homogeneous.row(3) = Eigen::VectorXd::Ones(W.rows());
    
    Eigen::MatrixXd ref_DD = ref2DD * W_homogeneous;
    Eigen::MatrixXd Cmm_pts = ref_DD.block(0, 0, 3, W.rows()).transpose();
    
    // Transform distance points
    Eigen::MatrixXd distpts_homogeneous(4, distpts.rows());
    distpts_homogeneous.block(0, 0, 3, distpts.rows()) = distpts.transpose();
    distpts_homogeneous.row(3) = Eigen::VectorXd::Ones(distpts.rows());
    
    Eigen::MatrixXd ref_dist_pts = ref2DD * distpts_homogeneous;
    
    return {Cmm_pts, ref_dist_pts};
}

std::pair<Eigen::Matrix<double, 3, 4>, std::vector<double>> Calibration::estimateCameraMatrix(
    const Eigen::MatrixXd& imagePoints,
    const Eigen::MatrixXd& worldPoints) {
    
    int M = worldPoints.rows();
    Eigen::VectorXd X = worldPoints.col(0);
    Eigen::VectorXd Y = worldPoints.col(1);
    Eigen::VectorXd Z = worldPoints.col(2);
    Eigen::VectorXd vec_1 = Eigen::VectorXd::Ones(M);
    Eigen::VectorXd vec_0 = Eigen::VectorXd::Zero(M);
    
    // Normalize image points
    Eigen::MatrixXd imagePoints_homogeneous(3, imagePoints.rows());
    imagePoints_homogeneous.block(0, 0, 2, imagePoints.rows()) = imagePoints.transpose();
    imagePoints_homogeneous.row(2) = Eigen::VectorXd::Ones(imagePoints.rows());
    
    auto [pts, T] = normalise2dpts(imagePoints_homogeneous);
    Eigen::Matrix3d Tinv = T.inverse();
    
    Eigen::VectorXd u = pts.row(0);
    Eigen::VectorXd v = pts.row(1);
    
    // Build A matrix
    Eigen::MatrixXd A(2 * M, 12);
    for (int i = 0; i < M; ++i) {
        A.row(2 * i) << X(i), Y(i), Z(i), 1, 0, 0, 0, 0, -u(i)*X(i), -u(i)*Y(i), -u(i)*Z(i), -u(i);
        A.row(2 * i + 1) << 0, 0, 0, 0, X(i), Y(i), Z(i), 1, -v(i)*X(i), -v(i)*Y(i), -v(i)*Z(i), -v(i);
    }
    
    // Solve using SVD
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::VectorXd P_vec = svd.matrixV().col(11); // Last column of V
    
    // Reshape to 3x4 matrix
    Eigen::Matrix<double, 3, 4> camMatrix;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            camMatrix(i, j) = P_vec(i * 4 + j);
        }
    }
    
    // Apply inverse normalization - Fix the matrix multiplication
    Eigen::Matrix<double, 3, 4> camMatrix_normalized;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            camMatrix_normalized(i, j) = 0.0;
            for (int k = 0; k < 3; ++k) {
                camMatrix_normalized(i, j) += Tinv(i, k) * camMatrix(k, j);
            }
        }
    }
    camMatrix = camMatrix_normalized;
    
    // Ensure positive determinant
    Eigen::Matrix3d M_left = camMatrix.block<3, 3>(0, 0);
    if (M_left.determinant() < 0) {
        camMatrix = -camMatrix;
    }
    
    // Calculate reprojection errors
    std::vector<double> reprojectionErrors;
    Eigen::MatrixXd worldPoints_homogeneous(worldPoints.rows(), 4);
    worldPoints_homogeneous.block(0, 0, worldPoints.rows(), 3) = worldPoints;
    worldPoints_homogeneous.col(3) = Eigen::VectorXd::Ones(worldPoints.rows());
    
    for (int i = 0; i < worldPoints.rows(); ++i) {
        Eigen::Vector4d worldPt = worldPoints_homogeneous.row(i);
        Eigen::Vector3d projectedPt = camMatrix * worldPt;
        projectedPt /= projectedPt(2);
        
        double error = std::sqrt(std::pow(imagePoints(i, 0) - projectedPt(0), 2) + 
                                std::pow(imagePoints(i, 1) - projectedPt(1), 2));
        reprojectionErrors.push_back(error);
    }
    
    return {camMatrix, reprojectionErrors};
}

std::tuple<Eigen::Matrix3d, Eigen::Matrix3d, Eigen::Vector3d, Eigen::Vector2d, Eigen::Vector3d> 
Calibration::decomposeCamera(const Eigen::Matrix<double, 3, 4>& P) {
    
    // Extract columns
    Eigen::Vector3d p1 = P.col(0);
    Eigen::Vector3d p2 = P.col(1);
    Eigen::Vector3d p3 = P.col(2);
    Eigen::Vector3d p4 = P.col(3);
    
    Eigen::Matrix3d M;
    M << p1, p2, p3;
    Eigen::Vector3d m3 = M.row(2);
    
    // Camera centre (analytic solution)
    double X = M.block<3, 2>(0, 1).determinant() * p4(0) - (M.col(0).cross(M.col(2))).dot(p4);
    double Y = -M.block<3, 2>(0, 0).determinant() * p4(1) + (M.col(1).cross(M.col(2))).dot(p4);
    double Z = M.block<3, 2>(0, 0).determinant() * p4(2) - (M.col(0).cross(M.col(1))).dot(p4);
    double T = -M.determinant();
    
    Eigen::Vector4d Pc_homogeneous(X, Y, Z, T);
    Pc_homogeneous /= Pc_homogeneous(3);
    Eigen::Vector3d Pc = Pc_homogeneous.head<3>();
    
    // Principal point
    Eigen::Vector3d pp_homogeneous = M * m3;
    pp_homogeneous /= pp_homogeneous(2);
    Eigen::Vector2d pp = pp_homogeneous.head<2>();
    
    // Principal vector
    Eigen::Vector3d pv = M.determinant() * m3;
    pv.normalize();
    
    // RQ decomposition
    auto [K, R] = rq3(M);
    
    return {K, R, Pc, pp, pv};
}

double Calibration::registrationCheck(
    const Eigen::Matrix<double, 3, 4>& P,
    const Eigen::MatrixXd& W_dist_pts,
    const std::string& position,
    double r,
    const std::string& path_d,
    int rewarp) {
    
    try {
        std::string folder = (rewarp == 1) ? "Rewarp\\" : "";
        
        // Read blob image and extract centroids
        std::string blob_file = path_d + "\\Output\\" + position + "\\PD\\" + position + "bw.png";
        cv::Mat blob = cv::imread(blob_file, cv::IMREAD_GRAYSCALE);
        
        if (blob.empty()) {
            std::cerr << "Warning: Could not read blob image: " << blob_file << std::endl;
            return 10.0;
        }
        
        // Extract centroids using OpenCV
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(blob, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        std::vector<cv::Point2f> centres;
        for (const auto& contour : contours) {
            cv::Moments m = cv::moments(contour);
            if (m.m00 != 0) {
                centres.push_back(cv::Point2f(m.m10 / m.m00, m.m01 / m.m00));
            }
        }
        
        // Project 3D points to 2D
        Eigen::MatrixXd projected_2d_pts = P * W_dist_pts;
        
        // Normalize by homogeneous coordinate
        for (int i = 0; i < projected_2d_pts.cols(); ++i) {
            projected_2d_pts.col(i) /= projected_2d_pts(2, i);
        }
        
        // Extract 2D coordinates and flip y-coordinate
        Eigen::MatrixXd projected_pts_2d(projected_2d_pts.cols(), 2);
        projected_pts_2d.col(0) = projected_2d_pts.row(0);
        projected_pts_2d.col(1) = r - projected_2d_pts.row(1).array();
        
        // Calculate distances between projected points and detected centres
        std::vector<double> error_in_pixels;
        
        for (int i = 0; i < projected_pts_2d.rows(); ++i) {
            double min_dist = std::numeric_limits<double>::max();
            
            for (const auto& centre : centres) {
                double dist = std::sqrt(std::pow(projected_pts_2d(i, 0) - centre.x, 2) + 
                                       std::pow(projected_pts_2d(i, 1) - centre.y, 2));
                min_dist = std::min(min_dist, dist);
            }
            
            error_in_pixels.push_back(min_dist);
        }
        
        // Calculate mean error
        double mean_error = 0.0;
        for (double error : error_in_pixels) {
            mean_error += error;
        }
        mean_error /= error_in_pixels.size();
        
        return mean_error;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in registration check: " << e.what() << std::endl;
        return 10.0;
    }
}

// Utility function implementations
Eigen::Matrix3d Calibration::quaternionToRotationMatrix(const std::vector<double>& q) {
    double w = q[0], x = q[1], y = q[2], z = q[3];
    
    Eigen::Matrix3d R;
    R << 1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y),
         2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x),
         2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y);
    
    return R;
}

std::pair<Eigen::MatrixXd, Eigen::Matrix3d> Calibration::normalise2dpts(const Eigen::MatrixXd& pts) {
    if (pts.rows() != 3) {
        throw std::invalid_argument("pts must be 3xN");
    }
    
    Eigen::MatrixXd newpts = pts;
    
    // Find finite points
    std::vector<int> finiteind;
    for (int i = 0; i < pts.cols(); ++i) {
        if (std::abs(pts(2, i)) > 1e-10) {
            finiteind.push_back(i);
        }
    }
    
    // Normalize finite points
    for (int idx : finiteind) {
        newpts(0, idx) /= newpts(2, idx);
        newpts(1, idx) /= newpts(2, idx);
        newpts(2, idx) = 1.0;
    }
    
    // Calculate centroid
    Eigen::Vector2d c = Eigen::Vector2d::Zero();
    for (int idx : finiteind) {
        c(0) += newpts(0, idx);
        c(1) += newpts(1, idx);
    }
    c /= finiteind.size();
    
    // Shift origin to centroid
    for (int idx : finiteind) {
        newpts(0, idx) -= c(0);
        newpts(1, idx) -= c(1);
    }
    
    // Calculate mean distance
    double meandist = 0.0;
    for (int idx : finiteind) {
        meandist += std::sqrt(newpts(0, idx)*newpts(0, idx) + newpts(1, idx)*newpts(1, idx));
    }
    meandist /= finiteind.size();
    
    double scale = std::sqrt(2.0) / meandist;
    
    // Create transformation matrix
    Eigen::Matrix3d T;
    T << scale, 0, -scale*c(0),
         0, scale, -scale*c(1),
         0, 0, 1;
    
    return {T * pts, T};
}

std::pair<Eigen::Matrix3d, Eigen::Matrix3d> Calibration::rq3(const Eigen::Matrix3d& A) {
    const double eps = 1e-10;
    Eigen::Matrix3d R = A;
    
    // Find rotation Qx to set A(3,2) to 0
    R(2, 2) += eps;
    double c = -R(2, 2) / std::sqrt(R(2, 2)*R(2, 2) + R(2, 1)*R(2, 1));
    double s = R(2, 1) / std::sqrt(R(2, 2)*R(2, 2) + R(2, 1)*R(2, 1));
    Eigen::Matrix3d Qx;
    Qx << 1, 0, 0,
          0, c, -s,
          0, s, c;
    R = R * Qx;
    
    // Find rotation Qy to set A(3,1) to 0
    R(2, 2) += eps;
    c = R(2, 2) / std::sqrt(R(2, 2)*R(2, 2) + R(2, 0)*R(2, 0));
    s = R(2, 0) / std::sqrt(R(2, 2)*R(2, 2) + R(2, 0)*R(2, 0));
    Eigen::Matrix3d Qy;
    Qy << c, 0, s,
          0, 1, 0,
          -s, 0, c;
    R = R * Qy;
    
    // Find rotation Qz to set A(2,1) to 0
    R(1, 1) += eps;
    c = -R(1, 1) / std::sqrt(R(1, 1)*R(1, 1) + R(1, 0)*R(1, 0));
    s = R(1, 0) / std::sqrt(R(1, 1)*R(1, 1) + R(1, 0)*R(1, 0));
    Eigen::Matrix3d Qz;
    Qz << c, -s, 0,
          s, c, 0,
          0, 0, 1;
    R = R * Qz;
    
    Eigen::Matrix3d Q = Qz.transpose() * Qy.transpose() * Qx.transpose();
    
    // Adjust R and Q so that diagonal elements of R are positive
    for (int n = 0; n < 3; ++n) {
        if (R(n, n) < 0) {
            R.col(n) = -R.col(n);
            Q.row(n) = -Q.row(n);
        }
    }
    
    return {R, Q};
}

Eigen::MatrixXd Calibration::readMatrix(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return Eigen::MatrixXd(0, 0);
    }
    
    std::vector<std::vector<double>> data;
    std::string line;
    
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<double> row;
        double value;
        
        while (iss >> value) {
            row.push_back(value);
        }
        
        if (!row.empty()) {
            data.push_back(row);
        }
    }
    
    if (data.empty()) {
        return Eigen::MatrixXd(0, 0);
    }
    
    int rows = data.size();
    int cols = data[0].size();
    
    Eigen::MatrixXd matrix(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix(i, j) = data[i][j];
        }
    }
    
    return matrix;
}

void Calibration::writeMatrix(const std::string& filename, const Eigen::MatrixXd& matrix, 
                             const std::string& delimiter, int precision) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    
    file << std::fixed << std::setprecision(precision);
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            file << matrix(i, j);
            if (j < matrix.cols() - 1) {
                file << delimiter;
            }
        }
        file << std::endl;
    }
}

void Calibration::appendMatrix(const std::string& filename, const Eigen::MatrixXd& matrix, 
                              const std::string& delimiter, int precision) {
    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for appending: " + filename);
    }
    
    file << std::fixed << std::setprecision(precision);
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            file << matrix(i, j);
            if (j < matrix.cols() - 1) {
                file << delimiter;
            }
        }
        file << std::endl;
    }
}

bool Calibration::fileExists(const std::string& filename) {
    return std::filesystem::exists(filename);
}

void Calibration::writeErrorLog(const std::string& message, const std::string& path_d, 
                               const std::string& position) {
    try {
        // Write to position-specific error log
        std::string outPath = path_d + "\\Output\\" + position;
        std::filesystem::create_directories(outPath);
        std::string fullFileName = outPath + "\\errorLogFile.txt";
        
        std::ofstream file(fullFileName);
        if (file.is_open()) {
            file << message << std::endl;
            file.close();
        }
        
        // Write to general error log
        std::ofstream generalFile("errorLogFile.txt", std::ios::app);
        if (generalFile.is_open()) {
            generalFile << message << std::endl;
            generalFile.close();
        }
        
        // Write to error store file
        std::string errorStoreFile = outPath + "\\errorStoreFile.txt";
        std::ofstream storeFile(errorStoreFile, std::ios::app);
        if (storeFile.is_open()) {
            storeFile << "Calibration Failure" << std::endl;
            storeFile << message << std::endl;
            storeFile.close();
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error writing log files: " << e.what() << std::endl;
    }
}

void Calibration::appLog(const std::string& operation, const std::string& path_d) {
    // Simple logging implementation
    std::cout << "Operation completed: " << operation << " in " << path_d << std::endl;
}

Eigen::MatrixXd Calibration::loadIcpPlatFid() {
    // This should load the equivalent of icpPlatFid.mat
    // For now, return an empty matrix - you'll need to implement this based on your data
    // You could load from a text file or implement a binary reader for .mat files
    
    // Try to load from a text file if available
    std::string icpFile = "icpPlatFid.txt";
    if (fileExists(icpFile)) {
        return readMatrix(icpFile);
    }
    
    // Return empty matrix if no data available
    return Eigen::MatrixXd(0, 0);
}