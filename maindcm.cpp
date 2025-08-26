#include <gdcmImageReader.h>
#include <gdcmImage.h>
#include <gdcmPixelFormat.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <string>
#include <json.hpp>
#include <filesystem>

#include "blob_detection.h"
#include "Center_detection.h"
#include "icp_angle.h"
#include "plate12icp.h"
#include "crop_image.h"
#include "plate12indexing.h"
#include "Calibration.h"

using namespace std;
using namespace cv;
using json = nlohmann::json;

cv::Mat readDICOM_GDCM(const std::string& filepath) {
    gdcm::ImageReader reader;
    reader.SetFileName(filepath.c_str());

    if (!reader.Read()) {
        throw std::runtime_error("Failed to read DICOM file: " + filepath);
    }

    const gdcm::Image& image = reader.GetImage();
    const unsigned int* dims = image.GetDimensions();
    int width = dims[0], height = dims[1];

    gdcm::PixelFormat pf = image.GetPixelFormat();
    int numChannels = pf.GetSamplesPerPixel();

    std::vector<char> buffer(image.GetBufferLength());
    if (!image.GetBuffer(buffer.data())) {
        throw std::runtime_error("Failed to get pixel buffer");
    }

    int cvDepth;
    switch (pf.GetScalarType()) {
        case gdcm::PixelFormat::UINT8:   cvDepth = CV_8U; break;
        case gdcm::PixelFormat::INT8:    cvDepth = CV_8S; break;
        case gdcm::PixelFormat::UINT16:  cvDepth = CV_16U; break;
        case gdcm::PixelFormat::INT16:   cvDepth = CV_16S; break;
        case gdcm::PixelFormat::UINT32:  cvDepth = CV_32S; break;
        case gdcm::PixelFormat::FLOAT32: cvDepth = CV_32F; break;
        case gdcm::PixelFormat::FLOAT64: cvDepth = CV_64F; break;
        default:
            throw std::runtime_error("Unsupported GDCM pixel format");
    }

    cv::Mat img(height, width, CV_MAKETYPE(cvDepth, numChannels), buffer.data());
    cv::Mat imgCopy = img.clone();

    double minVal, maxVal;
    cv::minMaxLoc(imgCopy, &minVal, &maxVal);
    cv::Mat img8U;
    imgCopy.convertTo(img8U, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

    return img8U;
}

vector<Point2f> extractCentroids(const Mat& img) {
    vector<Point2f> centroids;
    Mat labels, stats, centroidsMat;
    int n_labels = connectedComponentsWithStats(img, labels, stats, centroidsMat, 8, CV_32S);
    for (int i = 1; i < n_labels; ++i) {
        centroids.push_back(Point2f(
            static_cast<float>(centroidsMat.at<double>(i, 0)),
            static_cast<float>(centroidsMat.at<double>(i, 1))
        ));
    }
    return centroids;
}

// --- Draw a set of points ---
void drawPoints(Mat& img, const vector<Point2f>& points, Scalar color, int radius = 5) {
    for (const auto& pt : points) {
        if (pt.x == 0 && pt.y == 0) continue;
        circle(img, pt, radius, color, FILLED);
    }
}

// Function to perform calibration using JSON data
CalibrationResult performCalibrationFromJSON(const std::string& jsonPath, const std::string& path_d, double r) {
    try {
        // Read JSON file
        std::ifstream jsonFile(jsonPath);
        if (!jsonFile) {
            throw std::runtime_error("Cannot open JSON file: " + jsonPath);
        }
        
        json input_SSR;
        jsonFile >> input_SSR;
        
        // Extract position
        std::string position = input_SSR["Type"];
        
        // Extract C2R (Camera to Reference) transformation
        TransformationData C2R;
        C2R.tx = input_SSR["Marker_Reference"]["tx"];
        C2R.ty = input_SSR["Marker_Reference"]["ty"];
        C2R.tz = input_SSR["Marker_Reference"]["tz"];
        
        // Fix JSON to vector conversion
        auto rotation_json = input_SSR["Marker_Reference"]["Rotation"];
        C2R.rotation.clear();
        for (const auto& val : rotation_json) {
            C2R.rotation.push_back(val.get<double>());
        }
        
        // Extract C2D (Camera to Detector) transformation  
        TransformationData C2D;
        C2D.tx = input_SSR["Marker_DD"]["tx"];
        C2D.ty = input_SSR["Marker_DD"]["ty"];
        C2D.tz = input_SSR["Marker_DD"]["tz"];
        
        // Fix JSON to vector conversion
        auto rotation_json2 = input_SSR["Marker_DD"]["Rotation"];
        C2D.rotation.clear();
        for (const auto& val : rotation_json2) {
            C2D.rotation.push_back(val.get<double>());
        }
        
        // Extract CMM World Points (W matrix)
        auto cmm_world_points = input_SSR["CMM_WorldPoints"];
        int num_world_points = cmm_world_points.size();
        Eigen::MatrixXd W(num_world_points, 3);
        
        for (int i = 0; i < num_world_points; ++i) {
            W(i, 0) = cmm_world_points[i][0];  // x
            W(i, 1) = cmm_world_points[i][1];  // y
            W(i, 2) = cmm_world_points[i][2];  // z
        }
        
        // Extract CMM Distance Points (Dpts matrix)
        auto cmm_dist_points = input_SSR["CMM_Dist_pts"];
        int num_dist_points = cmm_dist_points.size();
        Eigen::MatrixXd Dpts(num_dist_points, 3);
        
        for (int i = 0; i < num_dist_points; ++i) {
            Dpts(i, 0) = cmm_dist_points[i][0];  // x
            Dpts(i, 1) = cmm_dist_points[i][1];  // y
            Dpts(i, 2) = cmm_dist_points[i][2];  // z
        }
        
        int rewarp = 0;  // Set to match MATLAB call
        
        std::cout << "CALIBRATION - Starting calibration process" << std::endl;
        std::cout << "-----------------------------" << std::endl;
        std::cout << "Position: " << position << std::endl;
        std::cout << "World Points: " << num_world_points << " points" << std::endl;
        std::cout << "Distance Points: " << num_dist_points << " points" << std::endl;
        
        // Perform calibration (equivalent to MATLAB call)
        CalibrationResult result = Calibration::calibrate(
            position, C2R, C2D, W, Dpts, r, path_d, rewarp
        );
        
        std::cout << "Calibration Error: " << result.RPE << std::endl;
        
        return result;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in JSON calibration: " << e.what() << std::endl;
        CalibrationResult errorResult;
        errorResult.success = false;
        errorResult.RPE = 10.0;
        errorResult.error_message = e.what();
        return errorResult;
    }
}

int main() {
    std::string output_dir = ".";
    std::string jsonPath = "C:\\Users\\vishw\\OneDrive\\Desktop\\data\\SCN 7\\inputLP.json";

    std::ifstream jsonFile(jsonPath);
    if (!jsonFile) {
        std::cerr << "Cannot open JSON: " << jsonPath << std::endl;
        return 1;
    }

    json input_SSR;
    jsonFile >> input_SSR;
    std::string dicomPath = input_SSR["Image"];
    std::string position = input_SSR["Type"];

    cv::Mat img8u;
    try {
        img8u = readDICOM_GDCM(dicomPath);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }

    SimpleBlobDetector::Params params;
    params.filterByArea = true;
    params.minArea = 15;
    params.maxArea = 900;
    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
    vector<KeyPoint> keypoints;
    detector->detect(img8u, keypoints);

    if (keypoints.size() < 0) {
        std::cerr << "Image quality check failed: Not enough blobs." << std::endl;
        return 1;
    }

    int xmin, ymin, xmax, ymax;

    // Crop directly in memory
    Mat cropped_input = CropImage::run(img8u, xmin, ymin, xmax, ymax);
    cv::imwrite("cropped_output.png", cropped_input);
    
    // Run blob detection directly on Mat
    BlobDetectionResult blobRes = run_blob_detection(cropped_input, output_dir);

    // Collect blob centers and radii
    std::vector<Point2f> centers;
    std::vector<float> radii;
    for (const auto& kp : blobRes.verifiedKeypoints) {
        centers.push_back(kp.pt);
        radii.push_back(kp.size / 2.0f);
    }

    std::vector<Point2f> centroids = extractCentroids(blobRes.binaryBlobImage);
    auto [C, first_ring_balls] = Center_detection(centroids, blobRes.binaryBlobImage.rows, blobRes.binaryBlobImage.cols);

    std::vector<std::vector<float>> Z1;
    for (size_t i = 0; i < centers.size(); ++i) {
        float r = radii[i];
        if (r >= 8.0f && r <= 30.0f) {
            float dist = norm(centers[i] - C);
            Z1.push_back({dist, centers[i].x, centers[i].y, r});
        }
    }

    std::vector<std::vector<float>> Z2 = Z1;
    std::sort(Z2.begin(), Z2.end(), [](const auto& a, const auto& b) { return a[0] < b[0]; });

    std::vector<std::vector<float>> Z2filt;
    for (const auto& row : Z2) {
        if (row[0] > 230 && row[0] < 450) Z2filt.push_back(row);
    }
    Z2 = Z2filt;

    std::vector<float> dists;
    for (const auto& row : Z2) dists.push_back(row[0]);
    float minDist = dists.empty() ? 0 : *min_element(dists.begin(), dists.end());
    float maxDist = dists.empty() ? 1 : *max_element(dists.begin(), dists.end());

    std::vector<float> Z2norm;
    for (const auto& d : dists) Z2norm.push_back((d - minDist) / (maxDist - minDist + 1e-6f));

    std::vector<size_t> Z2idx;
    for (size_t i = 0; i < Z2norm.size(); ++i)
        if (Z2norm[i] < 0.6) Z2idx.push_back(i);
    if (Z2idx.size() <= 1) for (size_t i = 0; i < Z2norm.size(); ++i)
        if (Z2norm[i] < 0.7) Z2idx.push_back(i);
    if (Z2idx.size() <= 1) for (size_t i = 0; i < Z2norm.size(); ++i)
        if (Z2norm[i] < 0.8) Z2idx.push_back(i);

    std::vector<std::vector<float>> Z3;
    for (auto idx : Z2idx) Z3.push_back(Z2[idx]);

    std::cout << "DEBUG: Z2 size: " << Z2.size() << std::endl;
    std::cout << "DEBUG: Z2idx size: " << Z2idx.size() << std::endl;
    std::cout << "DEBUG: Z3 size: " << Z3.size() << std::endl;
    

    std::ofstream icpFile("icp_2D.txt");
    for (const auto& row : Z3) {
        for (size_t i = 0; i < row.size(); ++i)
            icpFile << std::fixed << std::setprecision(6) << row[i] << (i + 1 < row.size() ? " " : "\n");
    }
    icpFile.close();

    std::vector<Point2f> icpFidVec;
    for (const auto& row : Z3)
        if (row.size() >= 3) icpFidVec.emplace_back(row[1], row[2]);

    PlateFiducials plateResult = plate12icp(blobRes.binaryBlobImage, C, icpFidVec);
    plateResult = plate12withICP_post(plateResult, Z3, C);

    // Save results
    std::ofstream resultFile(output_dir + "/final_plate_points.txt");
    resultFile << "# Plate 1 Points\n";
    for (const auto& pt : plateResult.final_plate1)
        resultFile << pt.x << " " << pt.y << " " << pt.label << "\n";

    resultFile << "\n# Plate 2 Points\n";
    for (const auto& pt : plateResult.final_plate2)
        resultFile << pt.x << " " << pt.y << " " << pt.label << "\n";

    resultFile << "\n# ICP Points\n";
    for (const auto& pt : plateResult.icpPlatFid)
        resultFile << pt.x << " " << pt.y << " " << pt.label << "\n";

    resultFile.close();

    // Create visualization on original cropped image with only numbers
    cv::Mat img_color;
    if (cropped_input.channels() == 1) {
        cv::cvtColor(cropped_input, img_color, cv::COLOR_GRAY2BGR);
    } else {
        img_color = cropped_input.clone();
    }

    // Draw all fiducial point numbers (Green text like MATLAB version)
    // Combine all points for unified numbering like MATLAB
    std::vector<cv::Point3f> allPoints; // x, y, label

    // Add all points to single vector
    for (const auto& pt : plateResult.final_plate1) {
        allPoints.push_back(cv::Point3f(pt.x, pt.y, pt.label));
    }
    for (const auto& pt : plateResult.final_plate2) {
        allPoints.push_back(cv::Point3f(pt.x, pt.y, pt.label));
    }
    for (const auto& pt : plateResult.icpPlatFid) {
        allPoints.push_back(cv::Point3f(pt.x, pt.y, pt.label));
    }

    // Draw numbers for all points (matching MATLAB style)
    for (const auto& pt : allPoints) {
        cv::Point2f center(pt.x, pt.y);
        std::string label = std::to_string(static_cast<int>(pt.z));
        
        // Calculate text size for centering
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
        cv::Point2f textPos = center - cv::Point2f(textSize.width/2.0f, -textSize.height/2.0f);
        
        // Green text (matching MATLAB TextColor='green')
        cv::putText(img_color, label, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    }

    // Save the image with index numbers
    std::string indexed_output = output_dir + "/indexed_fiducials.png";
    cv::imwrite(indexed_output, img_color);

    // ========================= CALIBRATION SECTION =========================
    std::cout << "\n=== STARTING CALIBRATION ===" << std::endl;
    
    // Configuration - equivalent to config.ErrorUB in MATLAB
    const double ERROR_UPPER_BOUND = 5.0;  // Adjust this value as needed
    
    int rewarp = 0;
    double r_param = static_cast<double>(img8u.rows);  // Use actual image height
    
    // appHead equivalent
    std::cout << "CALIBRATION" << std::endl;
    std::cout << "-----------------------------" << std::endl;
    
    // Extract calibration parameters from JSON
    TransformationData C2R, C2D;
    C2R.tx = input_SSR["Marker_Reference"]["tx"];
    C2R.ty = input_SSR["Marker_Reference"]["ty"];
    C2R.tz = input_SSR["Marker_Reference"]["tz"];
    
    // Fix JSON to vector conversion for C2R rotation
    auto c2r_rotation_json = input_SSR["Marker_Reference"]["Rotation"];
    C2R.rotation.clear();
    for (const auto& val : c2r_rotation_json) {
        C2R.rotation.push_back(val.get<double>());
    }
    
    C2D.tx = input_SSR["Marker_DD"]["tx"];
    C2D.ty = input_SSR["Marker_DD"]["ty"];
    C2D.tz = input_SSR["Marker_DD"]["tz"];
    
    // Fix JSON to vector conversion for C2D rotation
    auto c2d_rotation_json = input_SSR["Marker_DD"]["Rotation"];
    C2D.rotation.clear();
    for (const auto& val : c2d_rotation_json) {
        C2D.rotation.push_back(val.get<double>());
    }
    
    // Extract world points and distance points
    auto cmm_world_points = input_SSR["CMM_WorldPoints"];
    int num_world_points = cmm_world_points.size();
    Eigen::MatrixXd World(num_world_points, 3);
    
    for (int i = 0; i < num_world_points; ++i) {
        World(i, 0) = cmm_world_points[i][0];
        World(i, 1) = cmm_world_points[i][1];
        World(i, 2) = cmm_world_points[i][2];
    }
    
    auto cmm_dist_points = input_SSR["CMM_Dist_pts"];
    int num_dist_points = cmm_dist_points.size();
    Eigen::MatrixXd Dist_pts(num_dist_points, 3);
    
    for (int i = 0; i < num_dist_points; ++i) {
        Dist_pts(i, 0) = cmm_dist_points[i][0];
        Dist_pts(i, 1) = cmm_dist_points[i][1];
        Dist_pts(i, 2) = cmm_dist_points[i][2];
    }
    
    // Perform calibration (equivalent to MATLAB Calibration call)
    CalibrationResult calibResult = Calibration::calibrate(
        position, C2R, C2D, World, Dist_pts, r_param, output_dir, rewarp
    );
    
    std::cout << "Calibration Error: " << calibResult.RPE << std::endl;
    
    // Create output JSON structure (equivalent to output_SSR in MATLAB)
    json output_SSR = input_SSR;  // Start with input data
    
    // Check error threshold and proceed accordingly
    if (calibResult.RPE < ERROR_UPPER_BOUND) {
        std::cout << "Calibration successful - Error within bounds" << std::endl;
        
        // TODO: Add Imageinpaint equivalent here if needed
        // output_SSR = Imageinpaint(meta, position, output_SSR, path_d, rewarp);
        std::cout << "Imageinpaint - (Not implemented yet)" << std::endl;
        
        output_SSR["Status"] = "SUCCESS";
        output_SSR["ErrorMessage"] = "";
        output_SSR["RPE"] = calibResult.RPE;
        
        // Add calibration matrix to output
        std::vector<std::vector<double>> resultMatrix(4, std::vector<double>(4));
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                resultMatrix[i][j] = calibResult.Result_Matrix(i, j);
            }
        }
        output_SSR["Result_Matrix"] = resultMatrix;
        
    } else {
        std::cout << "Calibration failed - Error exceeds threshold" << std::endl;
        
        output_SSR["Status"] = "FAILURE";
        output_SSR["ErrorMessage"] = "Calibration failed";
        output_SSR["TwoD_Points"] = json::array();
        output_SSR["UD_InPaint_Image"] = "Empty";
        output_SSR["CropRoi"] = json::array();
        output_SSR["RPE"] = -1;
    }
    
    // Set additional output fields
    output_SSR["Type"] = position;
    output_SSR["RegistrationType"] = input_SSR.value("RegistrationType", "2D2D");
    output_SSR["Version"] = "1.0";  // Set your version
    
    // Write output JSON file
    std::string output_json_file = output_dir + "\\Output\\" + position + "\\output" + position + ".json";
    
    // Create directory if it doesn't exist
    std::filesystem::create_directories(output_dir + "\\Output\\" + position);
    
    std::ofstream outputFile(output_json_file);
    if (outputFile.is_open()) {
        outputFile << output_SSR.dump(4);  // Pretty print with 4-space indentation
        outputFile.close();
        std::cout << "Output JSON saved to: " << output_json_file << std::endl;
    } else {
        std::cerr << "Failed to write output JSON file: " << output_json_file << std::endl;
    }

    std::cout << "\n=== OUTPUT FILES ===" << std::endl;
    std::cout << "Original image with index numbers saved to: " << indexed_output << std::endl;
    std::cout << "Point coordinates saved to: " << output_dir + "/final_plate_points.txt" << std::endl;
    std::cout << "ICP 2D data saved to: icp_2D.txt" << std::endl;
    std::cout << "Calibration output saved to: " << output_json_file << std::endl;

    return 0;
}