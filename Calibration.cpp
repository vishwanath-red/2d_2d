#include "Calibration.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <numeric>
#include <random>
#include <unordered_set>
#include "rq3.h"
#include "normalise2dpts.h"
#include "decompose_camera.h"

CalibrationResult Calibration::calibrate(
    const std::string& position,
    const TransformationData& C2R,
    const TransformationData& C2D,
    const Eigen::MatrixXd& W,
    const Eigen::MatrixXd& Dpts,
    const std::string& path_d,
    int rewarp) {
    
    CalibrationResult result;
    result.success = false;
    result.RPE = 10.0;
    
    try {
        // Determine folder based on rewarp flag (matching MATLAB logic)
        std::string folder = (rewarp == 1) ? "Rewarp\\" : "";
        
        // Load ICP fiducial data (equivalent to MATLAB: load icpPlatFid.mat)
        Eigen::MatrixXd icpPlatFid = loadIcpPlatFid(path_d);
        
        // Use original W matrix (matching MATLAB where W = [W;icp40mmFid] is commented out)
        Eigen::MatrixXd W_working = W;
        
        // Reference conversion (matching MATLAB: [Ref_CMM_pts,Ref_dist_pts]=Ref_conversion(...))
        auto [Ref_CMM_pts, Ref_dist_pts] = refConversion(position, W_working, Dpts, C2R, C2D, path_d);
        
        // Read 2D points from file (matching maindcm.cpp path structure exactly)
        // maindcm.cpp saves to: output_dir + "\\Output\\" + position + "\\PD\\" + position + "_2D.txt"
        // where output_dir = "."
        std::string xy_file = path_d + "\\Output\\" + position + "\\PD\\" + position + "_2D.txt";
        // Optional MATLAB override for apples-to-apples comparison
       // std::string xy_override = path_d + "\\Output\\" + position + "\\PD\\" + position + "_2D_matlab.txt";
        // if (fileExists(xy_override)) {
        //     std::cout << "Using MATLAB-provided 2D points override: " << xy_override << std::endl;
        //     xy_file = xy_override;
        // }
        Eigen::MatrixXd xy_o = readMatrix(xy_file);
        
        if (xy_o.rows() == 0) {
            throw std::runtime_error("Failed to read 2D points file: " + xy_file);
        }
        
        // Enforce strict label-based ordering to match world-point order
        // Validate labels are integers and 1-based within Ref_CMM range
        for (int i = 0; i < xy_o.rows(); ++i) {
            double lbl = xy_o(i, 2);
            if (std::floor(lbl) != lbl || lbl < 1 || lbl > Ref_CMM_pts.rows()) {
                std::ostringstream oss;
                oss << "Invalid label at row " << i << ": " << lbl
                    << " (expected integer in [1," << Ref_CMM_pts.rows() << "])";
                throw std::runtime_error(oss.str());
            }
        }

       // Enforce MATLAB-equivalent sortrows(xy_o,'descend')
        // Build sortable rows: [x, y, label]
        struct Row { double x, y, label; };
        std::vector<Row> rows(xy_o.rows());
        for (int i = 0; i < xy_o.rows(); ++i) {
            rows[i] = {xy_o(i, 0), xy_o(i, 1), xy_o(i, 2)};
        }

        // Lexicographic sort: x desc, then y desc, then label desc
        std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) {
            if (a.x != b.x) return a.x > b.x;
            if (a.y != b.y) return a.y > b.y;
            return a.label > b.label;
        });

        // Rebuild xy_label_sorted and order (labels → 0-based)
        Eigen::MatrixXd xy_label_sorted(rows.size(), 3);
        Eigen::VectorXi order(rows.size());
        for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
            xy_label_sorted(i, 0) = rows[i].x;
            xy_label_sorted(i, 1) = rows[i].y;
            xy_label_sorted(i, 2) = rows[i].label;
            order(i) = static_cast<int>(rows[i].label) - 1;
        }

        
        // Prefer PD blob image height to match the image used for 2D detection
        double r = getImageHeightFromBlob(position, path_d);
        if (r <= 0) {
            // Fallback to cropped image
            r = getImageHeightFromCroppedImage(path_d);
            if (r <= 0) {
                // Final fallback to estimation from data
                r = getImageHeight(position, path_d, icpPlatFid);
                if (r <= 0) {
                    throw std::runtime_error("Could not determine image height from available sources");
                }
            }
        }
        
        // Extract xy coordinates and apply coordinate transformation 
        // (matching MATLAB: xy=[xy(:,1),r-xy(:,2)])
        Eigen::MatrixXd xy(xy_label_sorted.rows(), 2);
        xy.col(0) = xy_label_sorted.col(0);
        xy.col(1) = r - xy_label_sorted.col(1).array(); // Flip y-coordinate
        
        // Order Ref_CMM by labels
        Eigen::MatrixXd Ref_CMM_pts_order(order.size(), Ref_CMM_pts.cols());
        for (int i = 0; i < order.size(); ++i) {
            int lbl = order(i);
            if (lbl < 0 || lbl >= Ref_CMM_pts.rows()) {
                std::ostringstream oss;
                oss << "Label index out of bounds after sorting: " << lbl;
                throw std::runtime_error(oss.str());
            }
            Ref_CMM_pts_order.row(i) = Ref_CMM_pts.row(lbl);
        }



        // Estimate camera matrix using MATLAB-equivalent mapping
        auto [P0, reprojection_errors] = estimateCameraMatrix(xy, Ref_CMM_pts_order);
        if (!reprojection_errors.empty()) {
            double sum_err = 0.0, max_err = 0.0;
            for (double e : reprojection_errors) { sum_err += e; if (e > max_err) max_err = e; }
            double mean_err = sum_err / static_cast<double>(reprojection_errors.size());

            double final_mean = mean_err;
            double final_max = max_err;

            // Optional non-linear refinement to reduce reprojection error further
            if (mean_err > 1.0) {
                Eigen::MatrixXd img_h(3, xy.rows());
                img_h.block(0,0,2,xy.rows()) = xy.transpose();
                img_h.row(2) = Eigen::VectorXd::Ones(xy.rows());
                P0 = refineProjectionMatrix(P0, img_h, Ref_CMM_pts_order);
                // Recompute reprojection error after refinement
                Eigen::MatrixXd X_h(4, Ref_CMM_pts_order.rows());
                X_h.block(0,0,3,Ref_CMM_pts_order.rows()) = Ref_CMM_pts_order.transpose();
                X_h.row(3) = Eigen::VectorXd::Ones(Ref_CMM_pts_order.rows());
                double sum2 = 0.0, max2 = 0.0;
                for (int i = 0; i < Ref_CMM_pts_order.rows(); ++i) {
                    Eigen::Vector3d proj = P0 * X_h.col(i);
                    double u = proj(0)/proj(2);
                    double v = proj(1)/proj(2);
                    double dx = u - xy(i,0);
                    double dy = v - xy(i,1);
                    double e = std::sqrt(dx*dx + dy*dy);
                    sum2 += e; if (e > max2) max2 = e;
                }
                final_mean = sum2 / static_cast<double>(Ref_CMM_pts_order.rows());
                final_max = max2;
            }

            std::cout << "Reprojection error (px) - mean: " << final_mean << ", max: " << final_max << std::endl;
        }

        // Decompose camera matrix (matching MATLAB: [K,R_ct,Pc1, pp1, pv1] = decomposecamera(P0'))
        auto [K, R_ct, Pc1, pp1, pv1] = decomposeCamera(P0);
        // Normalize K to have K(2,2)=1 for consistency with MATLAB
        if (std::abs(K(2,2)) > 1e-12) {
            K /= K(2,2);
        }


        
        // Normalize K matrix (matching MATLAB: K_norm = K/K(3,3))
        Eigen::Matrix3d K_norm = K / K(2, 2);
        
        // Create Rt3 matrix (matching MATLAB: Rt3=[R_ct -R_ct*Pc1])
        Eigen::Matrix<double, 3, 4> Rt3;
        Rt3.block<3, 3>(0, 0) = R_ct;
        Rt3.block<3, 1>(0, 3) = -R_ct * Pc1;
        
        // Calculate normalized projection matrix (matching MATLAB: P_norm = K_norm*Rt3)
        Eigen::Matrix<double, 3, 4> P_norm = K_norm * Rt3;
        

        
        // Reshape to 1x12 vector (matching MATLAB: P_patient=reshape(P_norm',[1,12]))
        Eigen::VectorXd P_patient(12);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 4; ++j) {
                P_patient(i * 4 + j) = P_norm(i, j);
            }
        }
        
        // Write P_patient to position-specific file 
        // (matching MATLAB: dlmwrite([path_d,'\Output','\',position, folder, '\', 'P_Imf' position '.txt'],P_patient,...))
        std::string p_file = path_d + "\\Output\\" + position + folder + "\\P_Imf" + position + ".txt";
        
        // Ensure directory exists
        std::string p_dir = path_d + "\\Output\\" + position + folder;
        std::filesystem::create_directories(p_dir);
        
        writeMatrix(p_file, P_patient.transpose(), " ", 6);


        
        // Handle P_Imf.txt file writing based on position (matching MATLAB logic exactly)
        std::string main_p_file = path_d + "\\Output" + folder + "\\P_Imf.txt";
        
        // Ensure main output directory exists
        std::string main_dir = path_d + "\\Output" + folder;
        std::filesystem::create_directories(main_dir);
        
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
        
        // Registration check (matching MATLAB: Error=Registration_check(P0',Ref_dist_pts,position,r,path_d,rewarp))
        double error = registrationCheck(P0, Ref_dist_pts, position, r, path_d, rewarp);
        
        // Prepare result (matching MATLAB: js.Result_Matrix=[P0';0 0 0 1]; js.RPE = round(Error,3))
        result.Result_Matrix = Eigen::Matrix4d::Identity();
        result.Result_Matrix.block<3, 4>(0, 0) = P0;
        result.RPE = std::round(error * 1000.0) / 1000.0; // Round to 3 decimal places
        result.success = true;
        
        // Log success (matching MATLAB: appLog('CALIBRATION',path_d))
        appLog("CALIBRATION", path_d);
        
    } catch (const std::exception& e) {
        // Error handling matching MATLAB catch block
        result.success = false;
        result.RPE = 10.0;
        result.error_message = e.what();
        
        // Write error logs (matching MATLAB error logging)
        writeErrorLog("Calibration Failure: " + std::string(e.what()), path_d, position);
        
        std::cerr << "Calibration error: " << e.what() << std::endl;
    }
    
    return result;
}

double Calibration::getImageHeightFromCroppedImage(const std::string& path_d) {
    try {
        // Read the cropped image that maindcm.cpp saves
        // maindcm.cpp: cv::imwrite("cropped_output.png", cropped_input);
        std::string cropped_file = path_d + "\\cropped_output.png";
        cv::Mat cropped = cv::imread(cropped_file, cv::IMREAD_GRAYSCALE);
        if (!cropped.empty()) {
            return static_cast<double>(cropped.rows);
        }
        return -1.0; // Indicate failure
    } catch (const std::exception&) {
        return -1.0;
    }
}

double Calibration::getImageHeightFromBlob(const std::string& position, const std::string& path_d) {
    try {
        // Read blob image from EXACT path where maindcm.cpp saves it
        // maindcm.cpp: std::string bw_file = pd_dir + "\\" + position + "bw.png";
        // where pd_dir = output_dir + "\\Output\\" + position + "\\PD"
        std::string blob_file = path_d + "\\Output\\" + position + "\\PD\\" + position + "bw.png";
        cv::Mat blob = cv::imread(blob_file, cv::IMREAD_GRAYSCALE);
        if (!blob.empty()) {
            return static_cast<double>(blob.rows);
        }
        return -1.0; // Indicate failure
    } catch (const std::exception&) {
        return -1.0;
    }
}

double Calibration::getImageHeight(const std::string& position, const std::string& path_d, const Eigen::MatrixXd& icpPlatFid) {
    try {
        // Method 1: Use icpPlatFid data to estimate image bounds
        if (icpPlatFid.rows() > 0 && icpPlatFid.cols() >= 2) {
            double max_y = icpPlatFid.col(1).maxCoeff();
            double min_y = icpPlatFid.col(1).minCoeff();
            double estimated_height = max_y + (max_y - min_y) * 0.1; // Add 10% margin
            if (estimated_height > 100 && estimated_height < 10000) {
                return estimated_height;
            }
        }
        // Method 2: Try to read the original cropped image from EXACT path where maindcm.cpp saves it
        // maindcm.cpp: cv::imwrite("cropped_output.png", cropped_input);
        std::string cropped_file = path_d + "\\cropped_output.png";
        cv::Mat cropped = cv::imread(cropped_file, cv::IMREAD_GRAYSCALE);
        if (!cropped.empty()) {
            return static_cast<double>(cropped.rows);
        }
        // Method 3: Use 2D points file to estimate bounds from EXACT path where maindcm.cpp saves it
        // maindcm.cpp: std::string xy_file = pd_dir + "\\" + position + "_2D.txt";
        std::string xy_file = path_d + "\\Output\\" + position + "\\PD\\" + position + "_2D.txt";
        Eigen::MatrixXd xy_data = readMatrix(xy_file);
        if (xy_data.rows() > 0 && xy_data.cols() >= 2) {
            double max_y_2d = xy_data.col(1).maxCoeff();
            double min_y_2d = xy_data.col(1).minCoeff();
            double estimated_height_2d = max_y_2d + (max_y_2d - min_y_2d) * 0.1;
            if (estimated_height_2d > 100 && estimated_height_2d < 10000) {
                return estimated_height_2d;
            }
        }
        return 1024.0; // Default fallback value
    } catch (const std::exception&) {
        return 1024.0; // Default fallback value
    }
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
    // MATLAB: rot = quat2rotm([-q(4) q(1) q(2) q(3)]) where q = [x y z w]
    // Match MATLAB mapping exactly: [w,x,y,z] with negated w
    Eigen::Matrix3d rot = quaternionToRotationMatrix({ -C2R.rotation[3], C2R.rotation[0], C2R.rotation[1], C2R.rotation[2] });
    
    Eigen::Matrix4d cam2ref = Eigen::Matrix4d::Identity();
    cam2ref.block<3, 3>(0, 0) = rot;
    cam2ref.block<3, 1>(0, 3) = trans;
    
    // Detector marker transformation
    Eigen::Vector3d trans2(C2D.tx, C2D.ty, C2D.tz);
    Eigen::Matrix3d rot2 = quaternionToRotationMatrix({ -C2D.rotation[3], C2D.rotation[0], C2D.rotation[1], C2D.rotation[2] });
    
    Eigen::Matrix4d cam2DD = Eigen::Matrix4d::Identity();
    cam2DD.block<3, 3>(0, 0) = rot2;
    cam2DD.block<3, 1>(0, 3) = trans2;
    
    // Calculate transformations
    Eigen::Matrix4d ref2cam = cam2ref.inverse();
    Eigen::Matrix4d ref2DD = ref2cam * cam2DD;
    
    // Save ref2dd.txt for AP position
    if (position == "AP") {
        std::string ref2dd_dir = path_d + "\\Output\\" + position;
        std::filesystem::create_directories(ref2dd_dir);
        std::string ref2dd_file = ref2dd_dir + "\\ref2dd.txt";
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

    const int M = static_cast<int>(worldPoints.rows());
    if (M < 6 || imagePoints.rows() != M || imagePoints.cols() != 2 || worldPoints.cols() != 3) {
        return {Eigen::Matrix<double,3,4>::Zero(), std::vector<double>()};
    }

    // Helper: build DLT from a subset of indices, with normalised image points
    auto build_DLT = [&](const std::vector<int>& idx,
                         const Eigen::MatrixXd& img_norm,
                         const Eigen::Matrix3d& Tinv) -> Eigen::Matrix<double,3,4> {
        const int n = static_cast<int>(idx.size());
        Eigen::MatrixXd A(2*n, 12);
        for (int k = 0; k < n; ++k) {
            int i = idx[k];
            double X = worldPoints(i,0), Y = worldPoints(i,1), Z = worldPoints(i,2);
            double u = img_norm(0,i), v = img_norm(1,i);
            A.row(2*k)     << X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u;
            A.row(2*k + 1) << 0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v;
        }
        Eigen::MatrixXd AtA = A.transpose() * A;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(AtA);
        Eigen::Index minIndex; es.eigenvalues().minCoeff(&minIndex);
        Eigen::VectorXd P_vec = es.eigenvectors().col(minIndex);
        if (P_vec(11) < 0) P_vec = -P_vec;
        Eigen::Matrix<double,4,3> camMatrix_prime;
        camMatrix_prime << P_vec(0), P_vec(4), P_vec(8),
                           P_vec(1), P_vec(5), P_vec(9),
                           P_vec(2), P_vec(6), P_vec(10),
                           P_vec(3), P_vec(7), P_vec(11);
        camMatrix_prime = camMatrix_prime * Tinv.transpose();
        if (camMatrix_prime(3,2) < 0) camMatrix_prime = -camMatrix_prime;
        return camMatrix_prime.transpose(); // 3x4
    };

    // Prepare homogeneous and normalised image points
    Eigen::MatrixXd imagePoints_h(3, M);
    imagePoints_h.block(0,0,2,M) = imagePoints.transpose();
    imagePoints_h.row(2) = Eigen::VectorXd::Ones(M);
    auto [img_norm_h, T] = normalise2dpts(imagePoints_h);
    Eigen::Matrix3d Tinv = T.inverse();

    // RANSAC parameters
    const int s = 6;                 // minimal sample size (>=6 for stability)
    const int max_iters = 1000;
    const double inlier_thresh = 2.0; // px
    const double inlier_frac_stop = 0.8; // early stop if enough inliers

    std::mt19937 rng{std::random_device{}()};
    std::uniform_int_distribution<int> uni(0, M-1);

    int best_inliers = -1;
    Eigen::Matrix<double,3,4> best_P = Eigen::Matrix<double,3,4>::Zero();

    // Precompute normalized image arrays for quick access
    Eigen::VectorXd u = img_norm_h.row(0);
    Eigen::VectorXd v = img_norm_h.row(1);

    for (int iter = 0; iter < max_iters; ++iter) {
        // Draw unique sample indices
        std::vector<int> idx; idx.reserve(s);
        std::unordered_set<int> used;
        while ((int)idx.size() < s) {
            int r = uni(rng);
            if (used.insert(r).second) idx.push_back(r);
        }

        // Build model
        Eigen::Matrix<double,3,4> P = build_DLT(idx, img_norm_h, Tinv);
        if (!P.allFinite()) continue;

        // Count inliers using reprojection error
        int inliers = 0;
        for (int i = 0; i < M; ++i) {
            Eigen::Vector4d X(worldPoints(i,0), worldPoints(i,1), worldPoints(i,2), 1.0);
            Eigen::Vector3d p = P * X;
            double w = p(2);
            if (std::abs(w) < 1e-12) continue;
            double uu = p(0)/w, vv = p(1)/w;
            double du = imagePoints(i,0) - uu;
            double dv = imagePoints(i,1) - vv;
            double err = std::sqrt(du*du + dv*dv);
            if (err < inlier_thresh) ++inliers;
        }

        if (inliers > best_inliers) {
            best_inliers = inliers;
            best_P = P;
            if (inliers > inlier_frac_stop * M) break;
        }
    }

    // If no good model, fall back to all points DLT
    if (best_inliers <= 0) {
        std::vector<int> all_idx(M); std::iota(all_idx.begin(), all_idx.end(), 0);
        best_P = build_DLT(all_idx, img_norm_h, Tinv);
    } else {
        // Refit using inliers
        std::vector<int> inlier_idx; inlier_idx.reserve(best_inliers);
        for (int i = 0; i < M; ++i) {
            Eigen::Vector4d X(worldPoints(i,0), worldPoints(i,1), worldPoints(i,2), 1.0);
            Eigen::Vector3d p = best_P * X;
            double w = p(2);
            if (std::abs(w) < 1e-12) continue;
            double uu = p(0)/w, vv = p(1)/w;
            double du = imagePoints(i,0) - uu;
            double dv = imagePoints(i,1) - vv;
            double err = std::sqrt(du*du + dv*dv);
            if (err < inlier_thresh) inlier_idx.push_back(i);
        }
        if ((int)inlier_idx.size() >= 6) {
            best_P = build_DLT(inlier_idx, img_norm_h, Tinv);
        }
    }

    // Non-linear refinement on inliers for maximum accuracy
    // Build imagePoints_h for refine using original (not normalized) coordinates
    Eigen::MatrixXd img_h(3, M);
    img_h.block(0,0,2,M) = imagePoints.transpose();
    img_h.row(2) = Eigen::VectorXd::Ones(M);
    best_P = refineProjectionMatrix(best_P, img_h, worldPoints);

    // Compute final reprojection errors
    std::vector<double> reprojectionErrors; reprojectionErrors.reserve(M);
    for (int i = 0; i < M; ++i) {
        Eigen::Vector4d X(worldPoints(i,0), worldPoints(i,1), worldPoints(i,2), 1.0);
        Eigen::Vector3d p = best_P * X;
        double w = p(2);
        if (std::abs(w) < 1e-12) { reprojectionErrors.push_back(1e6); continue; }
        double uu = p(0)/w, vv = p(1)/w;
        double du = imagePoints(i,0) - uu;
        double dv = imagePoints(i,1) - vv;
        reprojectionErrors.push_back(std::sqrt(du*du + dv*dv));
    }

    return {best_P, reprojectionErrors};
}

std::tuple<Eigen::Matrix3d, Eigen::Matrix3d, Eigen::Vector3d, Eigen::Vector2d, Eigen::Vector3d> 
Calibration::decomposeCamera(const Eigen::Matrix<double, 3, 4>& P) {
    // Delegate to header-only helper for clarity and testability
    return decompose_camera(P);
}

// Convert quaternion [w, x, y, z] to rotation matrix (normalized)
Eigen::Matrix3d Calibration::quaternionToRotationMatrix(const std::vector<double>& q) {
    if (q.size() != 4) throw std::invalid_argument("Quaternion must have 4 elements [w,x,y,z]");
    double w = q[0], x = q[1], y = q[2], z = q[3];
    double n = std::sqrt(w*w + x*x + y*y + z*z);
    if (n < 1e-12) throw std::invalid_argument("Quaternion norm is zero");
    w /= n; x /= n; y /= n; z /= n;
    Eigen::Matrix3d R;
    R << 1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w),
             2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w),
             2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y);
    return R;
}

// Adapter to header normalisation function
std::pair<Eigen::MatrixXd, Eigen::Matrix3d> Calibration::normalise2dpts(const Eigen::MatrixXd& pts) {
    return ::normalise2dpts(pts);
}

// File utilities
bool Calibration::fileExists(const std::string& filename) {
    return std::filesystem::exists(filename);
}

Eigen::MatrixXd Calibration::readMatrix(const std::string& filename) {
    std::ifstream ifs(filename);
    if (!ifs.is_open()) return Eigen::MatrixXd();
    std::vector<std::vector<double>> rows;
    std::string line;
    size_t cols = 0;
    while (std::getline(ifs, line)) {
        std::stringstream ss(line);
        std::vector<double> values;
        double v;
        while (ss >> v) values.push_back(v);
        if (!values.empty()) {
            if (cols == 0) cols = values.size();
            rows.push_back(std::move(values));
        }
    }
    if (rows.empty() || cols == 0) return Eigen::MatrixXd();
    Eigen::MatrixXd M(rows.size(), cols);
    for (size_t i = 0; i < rows.size(); ++i)
        for (size_t j = 0; j < cols; ++j)
            M(i, j) = rows[i][j];
    return M;
}

void Calibration::writeMatrix(const std::string& filename, const Eigen::MatrixXd& matrix, 
                              const std::string& delimiter, int precision) {
    std::filesystem::create_directories(std::filesystem::path(filename).parent_path());
    std::ofstream ofs(filename);
    ofs << std::fixed << std::setprecision(precision);
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            ofs << matrix(i, j);
            if (j + 1 < matrix.cols()) ofs << delimiter;
        }
        ofs << '\n';
    }
}

void Calibration::appendMatrix(const std::string& filename, const Eigen::MatrixXd& matrix, 
                               const std::string& delimiter, int precision) {
    std::filesystem::create_directories(std::filesystem::path(filename).parent_path());
    std::ofstream ofs(filename, std::ios::app);
    ofs << std::fixed << std::setprecision(precision);
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            ofs << matrix(i, j);
            if (j + 1 < matrix.cols()) ofs << delimiter;
        }
        ofs << '\n';
    }
}

void Calibration::writeErrorLog(const std::string& message, const std::string& path_d, 
                                const std::string& position) {
    try {
        std::string outPath = path_d + "\\Output\\" + position;
        std::filesystem::create_directories(outPath);
        // errorLogFile.txt (overwrite)
        {
            std::ofstream fid(outPath + "\\errorLogFile.txt");
            fid << message << '\n';
        }
        // errorStoreFile.txt (append)
        {
            std::ofstream fid(outPath + "\\errorStoreFile.txt", std::ios::app);
            fid << "Calibration Failure\n" << message << '\n';
        }
    } catch (...) {}
}

void Calibration::appLog(const std::string& operation, const std::string& path_d) {
    try {
        std::string dir = path_d + "\\MatlabAppLog";
        std::filesystem::create_directories(dir);
        std::ofstream fid(dir + "\\appLog.txt", std::ios::app);
        auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        fid << operation << " at " << std::put_time(std::localtime(&now), "%Y-%m-%d %H:%M:%S") << '\n';
    } catch (...) {}
}

Eigen::MatrixXd Calibration::loadIcpPlatFid(const std::string& path_d) {
    try {
        std::string file = path_d + "\\icpPlatFid.txt";
        if (!fileExists(file)) return Eigen::MatrixXd();
        Eigen::MatrixXd M = readMatrix(file);
        // Expect columns: x y label; keep as-is
        return M;
    } catch (...) {
        return Eigen::MatrixXd();
    }
}

// Non-linear refinement using Gauss-Newton on reprojection error
// imagePoints_homogeneous: 3xN (u,v,1)
// worldPoints: Nx3
Eigen::Matrix<double, 3, 4> Calibration::refineProjectionMatrix(
    const Eigen::Matrix<double, 3, 4>& P_init,
    const Eigen::MatrixXd& imagePoints_homogeneous,
    const Eigen::MatrixXd& worldPoints)
{
    Eigen::Matrix<double, 3, 4> P = P_init;
    const int N = static_cast<int>(worldPoints.rows());
    if (N < 6) return P; // need enough points

    // Build homogeneous 3D points (4xN)
    Eigen::MatrixXd X_h(4, N);
    X_h.block(0,0,3,N) = worldPoints.transpose();
    X_h.row(3) = Eigen::VectorXd::Ones(N);

    const int max_iters = 50;
    const double eps = 1e-8;

    double lambda = 1e-3; // LM damping (persist across iterations)

    for (int it = 0; it < max_iters; ++it) {
        // Residuals (2N) and Jacobian (2N x 12)
        Eigen::VectorXd r(2*N);
        Eigen::MatrixXd J(2*N, 12);
        J.setZero();

        std::vector<double> res_norms; res_norms.reserve(N);

        for (int i = 0; i < N; ++i) {
            Eigen::Vector4d X = X_h.col(i);
            Eigen::Vector3d proj = P * X;
            double w = proj(2);
            if (std::abs(w) < 1e-12) { r(2*i)=r(2*i+1)=0; continue; }
            double u = proj(0) / w;
            double v = proj(1) / w;
            double u_obs = imagePoints_homogeneous(0, i);
            double v_obs = imagePoints_homogeneous(1, i);
            double du = u - u_obs;
            double dv = v - v_obs;
            r(2*i)   = du;
            r(2*i+1) = dv;
            res_norms.push_back(std::sqrt(du*du + dv*dv));

            // Derivatives wrt P entries: p00..p02,p03, p10..p12,p13, p20..p22,p23
            Eigen::RowVector4d Xt = X.transpose();
            double invw = 1.0 / w;
            for (int k = 0; k < 4; ++k) {
                double Xk = X(k);
                // row for u
                J(2*i, 0 + k)  = Xk * invw;          // p0k
                J(2*i, 8 + k)  = -u * invw * Xk;     // p2k
                // row for v
                J(2*i+1, 4 + k) = Xk * invw;         // p1k
                J(2*i+1, 8 + k) = -v * invw * Xk;    // p2k
            }
        }

        // Robust IRLS weights (Huber)
        auto median = [](std::vector<double> v){
            if (v.empty()) return 0.0; size_t n=v.size();
            std::nth_element(v.begin(), v.begin()+n/2, v.end());
            double m = v[n/2];
            if (n%2==0){ auto mx=*std::max_element(v.begin(), v.begin()+n/2); m=(m+mx)/2.0; }
            return m;
        };
        double med = median(res_norms);
        for (double &x : res_norms) x = std::abs(x - med);
        double mad = median(res_norms);
        double s = std::max(1e-6, 1.4826 * mad); // robust scale
        const double delta = 1.5; // Huber threshold in pixels

        Eigen::VectorXd wts(2*N); wts.setOnes();
        for (int i = 0; i < N; ++i) {
            double ri = std::sqrt(r(2*i)*r(2*i) + r(2*i+1)*r(2*i+1));
            double t = ri / s;
            double wi = (t <= delta) ? 1.0 : (delta / t);
            wts(2*i) = wi; wts(2*i+1) = wi;
        }

        // Form weighted normal equations (LM): (J^T W J + lambda I) dp = - J^T W r
        Eigen::MatrixXd W = wts.asDiagonal();
        Eigen::MatrixXd H = J.transpose() * W * J + lambda * Eigen::MatrixXd::Identity(12,12);
        Eigen::VectorXd g = - J.transpose() * W * r;
        Eigen::VectorXd dp = H.ldlt().solve(g);
        if (!dp.allFinite()) break;
        if (dp.norm() < eps) break;

        // Update P
        Eigen::Matrix<double, 3, 4> dP;
        dP << dp(0), dp(1), dp(2), dp(3),
              dp(4), dp(5), dp(6), dp(7),
              dp(8), dp(9), dp(10), dp(11);

        Eigen::Matrix<double, 3, 4> P_new = P + dP;

        // Evaluate new weighted cost; accept if better
        double old_cost = (W * r).squaredNorm();
        Eigen::VectorXd r_new(2*N);
        for (int i = 0; i < N; ++i) {
            Eigen::Vector3d proj = P_new * X_h.col(i);
            double w = proj(2);
            if (std::abs(w) < 1e-12) { r_new(2*i)=r_new(2*i+1)=0; continue; }
            double u = proj(0) / w;
            double v = proj(1) / w;
            double u_obs = imagePoints_homogeneous(0, i);
            double v_obs = imagePoints_homogeneous(1, i);
            r_new(2*i)   = u - u_obs;
            r_new(2*i+1) = v - v_obs;
        }
        double new_cost = (W * r_new).squaredNorm();
        if (new_cost < old_cost) {
            P = P_new;
            lambda = std::max(lambda * 0.7, 1e-6);
        } else {
            lambda = std::min(lambda * 2.0, 1e3);
        }
    }

    return P;
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
        
        // Read blob image generated by C++ pipeline (SimpleBlobDetector output). Ignore MATLAB overrides for consistency.
        std::string blob_file = path_d + "\\Output\\" + position + "\\PD\\" + position + "bw.png";
        cv::Mat blob = cv::imread(blob_file, cv::IMREAD_GRAYSCALE);

        if (blob.empty()) {
            std::cerr << "Warning: Could not read blob image: " << blob_file << std::endl;
            return 10.0;
        }
        
        // Ensure binary image similar to MATLAB logical image: treat any >0 as foreground
        cv::threshold(blob, blob, 0, 255, cv::THRESH_BINARY);
        
        // Use connected components with stats to match MATLAB regionprops('Centroid') pixel-based centroids
        const int connectivity = 8; // MATLAB uses 8-connectivity for 2D by default
        cv::Mat labels, stats, centroidsMat;
        int nLabels = cv::connectedComponentsWithStats(blob, labels, stats, centroidsMat, connectivity, CV_32S);
        
        // Collect all component centroids except background (label 0), no area/border filtering to match MATLAB
        std::vector<cv::Point2f> centres;
        centres.reserve(std::max(nLabels - 1, 0));
        for (int i = 1; i < nLabels; ++i) { // skip background = 0
            double cx = centroidsMat.at<double>(i, 0);
            double cy = centroidsMat.at<double>(i, 1);
            centres.emplace_back(static_cast<float>(cx), static_cast<float>(cy));
        }
        
        if (centres.empty()) {
            std::cerr << "Warning: No centres detected in blob image." << std::endl;
            return 10.0;
        }
        
        // Project 3D points to 2D (homogeneous), then normalize
        Eigen::MatrixXd projected_2d_pts = P * W_dist_pts; // 3xN
        for (int i = 0; i < projected_2d_pts.cols(); ++i) {
            projected_2d_pts.col(i) /= projected_2d_pts(2, i);
        }
        
        // Extract 2D coordinates and flip y-coordinate to match MATLAB using provided r (consistent with MATLAB script)
        const int N = static_cast<int>(projected_2d_pts.cols());
        Eigen::MatrixXd projected_pts_2d(N, 2);
        projected_pts_2d.col(0) = projected_2d_pts.row(0);
        projected_pts_2d.col(1) = r - projected_2d_pts.row(1).array();
        
        // Build distance matrix (pixel distances between projected points and blob centres)
        Eigen::MatrixXd distance_compute(N, centres.size());
        for (int i = 0; i < N; ++i) {
            const double px = projected_pts_2d(i, 0);
            const double py = projected_pts_2d(i, 1);
            for (size_t j = 0; j < centres.size(); ++j) {
                const double dx = px - centres[j].x;
                const double dy = py - centres[j].y;
                distance_compute(i, j) = std::sqrt(dx*dx + dy*dy);
            }
        }

        // Compute per-row nearest neighbor distances and indices
        std::vector<int> rowMinIdx(N, -1);
        std::vector<double> rowMinDist(N, std::numeric_limits<double>::infinity());
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < distance_compute.cols(); ++j) {
                double d = distance_compute(i, j);
                if (d < rowMinDist[i]) { rowMinDist[i] = d; rowMinIdx[i] = j; }
            }
        }

        // Robust stats: median and MAD of nearest distances
        auto median_of = [](std::vector<double> v) {
            if (v.empty()) return 0.0; 
            size_t n = v.size();
            std::nth_element(v.begin(), v.begin() + n/2, v.end());
            double m = v[n/2];
            if (n % 2 == 0) {
                auto max_it = std::max_element(v.begin(), v.begin() + n/2);
                m = (m + *max_it) / 2.0;
            }
            return m;
        };

        std::vector<double> nn = rowMinDist; // copy
        double med = median_of(nn);
        // MAD
        for (double &x : nn) x = std::abs(x - med);
        double mad_raw = median_of(nn);
        double sigma = 1.4826 * mad_raw; // robust std estimate

        // Threshold bounds and sweep step (px)
        const double t_min = 0.1;
        const double t_max = 1.5;
        const double t_step = 0.1;

        // Column-wise nearest neighbors for mutual check
        std::vector<int> colMinIdx(distance_compute.cols(), -1);
        std::vector<double> colMinDist(distance_compute.cols(), std::numeric_limits<double>::infinity());
        for (int j = 0; j < distance_compute.cols(); ++j) {
            for (int i = 0; i < N; ++i) {
                double d = distance_compute(i, j);
                if (d < colMinDist[j]) { colMinDist[j] = d; colMinIdx[j] = i; }
            }
        }

        // Precompute fallback (mean per-row minima)
        double sum_min = 0.0;
        for (int i = 0; i < N; ++i) sum_min += rowMinDist[i];
        double fallback_err = sum_min / static_cast<double>(N);

        // Sweep thresholds and pick the one yielding lowest error
        double best_err = std::numeric_limits<double>::infinity();
        double best_t = t_min;
        int steps = static_cast<int>(std::floor((t_max - t_min) / t_step + 1e-9));
        for (int s = 0; s <= steps; ++s) {
            double t = t_min + s * t_step;
            std::vector<double> matchedDistances;
            matchedDistances.reserve(N);
            for (int i = 0; i < N; ++i) {
                int j = rowMinIdx[i];
                if (j >= 0 && colMinIdx[j] == i && rowMinDist[i] <= t) {
                    matchedDistances.push_back(rowMinDist[i]);
                }
            }

            if (!matchedDistances.empty()) {
                // Trimmed mean (drop top 10% to reduce outliers)
                std::sort(matchedDistances.begin(), matchedDistances.end());
                size_t k = matchedDistances.size();
                size_t drop = (k >= 10) ? static_cast<size_t>(std::floor(0.1 * k)) : 0;
                double sum = 0.0; size_t cnt = 0;
                for (size_t idx = 0; idx < k - drop; ++idx) { sum += matchedDistances[idx]; ++cnt; }
                double err = (cnt > 0) ? (sum / static_cast<double>(cnt)) : matchedDistances.back();
                if (err < best_err) { best_err = err; best_t = t; }
            }
        }

        // Hungarian 1-1 assignment (robust) using best threshold found
        auto hungarianAssign = [&](const Eigen::MatrixXd& cost) -> std::vector<int> {
            const int n = static_cast<int>(cost.rows());
            std::vector<double> u(n + 1, 0.0), v(n + 1, 0.0);
            std::vector<int> p(n + 1, 0), way(n + 1, 0);
            for (int i = 1; i <= n; ++i) {
                p[0] = i;
                int j0 = 0; 
                std::vector<double> minv(n + 1, std::numeric_limits<double>::infinity());
                std::vector<char> used(n + 1, false);
                do {
                    used[j0] = true;
                    int i0 = p[j0], j1 = 0;
                    double delta = std::numeric_limits<double>::infinity();
                    for (int j = 1; j <= n; ++j) if (!used[j]) {
                        double cur = cost(i0 - 1, j - 1) - u[i0] - v[j];
                        if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                        if (minv[j] < delta) { delta = minv[j]; j1 = j; }
                    }
                    for (int j = 0; j <= n; ++j) {
                        if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                        else { minv[j] -= delta; }
                    }
                    j0 = j1;
                } while (p[j0] != 0);
                do { int j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
            }
            std::vector<int> row_to_col(n, -1);
            for (int j = 1; j <= n; ++j) if (p[j] > 0) row_to_col[p[j] - 1] = j - 1;
            return row_to_col;
        };

        // Build augmented square cost matrix to allow unmatched rows/columns via dummies
        const int Rn = N; // rows = projected points
        const int Cn = static_cast<int>(centres.size()); // cols = blob centres
        const double t = best_t; // gating threshold from sweep
        const double INF = 1e9;
        const double dummyPenalty = std::max(0.75 * t, 0.5); // cost to leave unmatched
        const int sz = Rn + Cn; // augmented square
        Eigen::MatrixXd AC(sz, sz);
        AC.setZero();
        // Top-left: real costs with gating
        for (int i = 0; i < Rn; ++i) {
            for (int j = 0; j < Cn; ++j) {
                double c = distance_compute(i, j);
                if (!(c <= t)) c = 10.0 * t; // penalize invalid matches
                AC(i, j) = c;
            }
        }
        // Top-right: dummy columns for rows (leave row unmatched)
        for (int i = 0; i < Rn; ++i) {
            for (int j = 0; j < Rn; ++j) {
                AC(i, Cn + j) = dummyPenalty;
            }
        }
        // Bottom-left: dummy rows for columns (leave column unmatched)
        for (int i = 0; i < Cn; ++i) {
            for (int j = 0; j < Cn; ++j) {
                AC(Rn + i, j) = dummyPenalty;
            }
        }
        // Bottom-right: zeros
        // Solve assignment
        std::vector<int> row_to_col = hungarianAssign(AC);

        // Collect matched real pairs under threshold
        std::vector<double> matchedDistances;
        matchedDistances.reserve(std::min(Rn, Cn));
        for (int i = 0; i < Rn; ++i) {
            int j = row_to_col[i];
            if (j >= 0 && j < Cn) {
                double d = distance_compute(i, j);
                if (d <= t) matchedDistances.push_back(d);
            }
        }

        double hung_err = std::numeric_limits<double>::infinity();
        if (!matchedDistances.empty()) {
            std::sort(matchedDistances.begin(), matchedDistances.end());
            size_t k = matchedDistances.size();
            size_t drop = (k >= 10) ? static_cast<size_t>(std::floor(0.1 * k)) : 0;
            double sum = 0.0; size_t cnt = 0;
            for (size_t idx = 0; idx < k - drop; ++idx) { sum += matchedDistances[idx]; ++cnt; }
            if (cnt > 0) hung_err = sum / static_cast<double>(cnt);
        }

        // Prefer Hungarian result if valid; otherwise fall back to mutual-NN result; else fallback
        if (std::isfinite(hung_err)) {
            return std::min(hung_err, best_err);
        }
        if (std::isfinite(best_err)) {
            return best_err;
        }
        return fallback_err;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in registration check: " << e.what() << std::endl;
        return 10.0;
    }
}

// Utility function implementations
// (All utility functions are defined above; this block intentionally left empty to avoid duplicates.)