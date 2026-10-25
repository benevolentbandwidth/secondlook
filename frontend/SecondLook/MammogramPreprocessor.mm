#import <opencv2/opencv.hpp>
#import <opencv2/imgproc.hpp>
#import <opencv2/imgcodecs/ios.h>
#import "MammogramPreprocessor.h"

@implementation MammogramPreprocessor

// --- _imaging_utils.breast_mask: Otsu → largest connected component
// --- → morphological close (15x15 ellipse). Exact port.
static cv::Mat breastMask(const cv::Mat &gray) {
    cv::Mat binary;
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);

    cv::Mat labels, stats, centroids;
    int numLabels = cv::connectedComponentsWithStats(binary, labels, stats,
                                                     centroids, 8);
    if (numLabels < 2) return binary;  // fallback, same as Python

    // Label 0 is background; find largest foreground component.
    int largest = 1;
    int maxArea = 0;
    for (int i = 1; i < numLabels; i++) {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area > maxArea) { maxArea = area; largest = i; }
    }
    cv::Mat mask = (labels == largest);  // 255 where true, CV_8U

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                               cv::Size(15, 15));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    return mask;
}

// --- preprocessor._find_pectoral_line ---
static bool findPectoralLine(const std::vector<cv::Vec4i> &lines, int roiH,
                             cv::Vec4i &best) {
    double bestScore = -1;
    bool found = false;
    for (const auto &l : lines) {
        double dx = l[2] - l[0], dy = l[3] - l[1];
        double length = std::hypot(dx, dy);
        if (dx == 0) continue;
        double slope = dy / dx;
        if (slope <= 0.3 || slope > 5.0) continue;
        int topY = std::min(l[1], l[3]);
        if (topY > roiH * 0.3) continue;
        if (length > bestScore) { bestScore = length; best = l; found = true; }
    }
    return found;
}

// --- preprocessor._remove_pectoral ---
static cv::Mat removePectoral(const cv::Mat &gray) {
    int h = gray.rows, w = gray.cols;
    int roiH = (int)(h * 0.4);
    cv::Mat roi = gray(cv::Rect(0, 0, w, roiH));

    cv::Mat edges;
    cv::Canny(roi, edges, 30, 100);

    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(edges, lines, 1, CV_PI / 180, 50, roiH / 4.0, 20);

    cv::Mat result = gray.clone();
    if (lines.empty()) return result;

    cv::Vec4i line;
    if (!findPectoralLine(lines, roiH, line)) return result;

    int x1 = line[0], y1 = line[1], x2 = line[2], y2 = line[3];
    cv::Mat pecMask = cv::Mat::zeros(h, w, CV_8UC1);
    std::vector<cv::Point> pts = { {0, 0}, {x2, y2}, {x1, y1}, {0, y1} };
    cv::fillPoly(pecMask, std::vector<std::vector<cv::Point>>{pts},
                 cv::Scalar(255));

    // False-positive guard: skip if wedge covers > 25% of image (same as Python)
    double frac = (double)cv::countNonZero(pecMask) / (pecMask.rows * pecMask.cols);
    if (frac > 0.25) return result;

    result.setTo(0, pecMask);
    return result;
}

// --- preprocessor._normalize_orientation ---
// Centroid-x computed manually (equivalent to cv::moments m10/m00).
static cv::Mat normalizeOrientation(const cv::Mat &gray, const cv::Mat &mask) {
    cv::Mat colSum;
    cv::reduce(mask, colSum, 0, cv::REDUCE_SUM, CV_64F);  // 1 x w column sums

    double m00 = 0, m10 = 0;
    for (int x = 0; x < colSum.cols; x++) {
        double v = colSum.at<double>(0, x);
        m00 += v;
        m10 += v * x;
    }
    if (m00 == 0) return gray;  // cannot determine orientation — leave as-is

    double cx = m10 / m00;
    if (cx < gray.cols / 2.0) {
        cv::Mat flipped;
        cv::flip(gray, flipped, 1);
        return flipped;
    }
    return gray;
}

+ (nullable NSData *)preprocess:(UIImage *)image size:(int)size {
    if (image == nil) return nil;

    cv::Mat raw;
    UIImageToMat(image, raw);
    if (raw.empty()) return nil;

    // 1. Grayscale (UIImageToMat yields RGBA; RGBA2GRAY uses the same luma
    //    weights as Python's BGR2GRAY on BGR data — results match)
    cv::Mat gray;
    if (raw.channels() == 4)      cv::cvtColor(raw, gray, cv::COLOR_RGBA2GRAY);
    else if (raw.channels() == 3) cv::cvtColor(raw, gray, cv::COLOR_RGB2GRAY);
    else                          gray = raw;

    // 2. CLAHE, clipLimit=2.0, tileGrid 8x8 — same params as _apply_clahe.
    //    UIImage input is 8-bit, so the uint16→uint8 renorm branch never runs.
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    cv::Mat enhanced;
    clahe->apply(gray, enhanced);

    // 3. Breast mask + bitwise_and
    cv::Mat mask = breastMask(enhanced);
    cv::Mat masked;
    cv::bitwise_and(enhanced, enhanced, masked, mask);

    // 4. Pectoral removal
    cv::Mat noPec = removePectoral(masked);

    // 5. Orientation normalization
    cv::Mat oriented = normalizeOrientation(noPec, mask);

    // 6. Resize (INTER_AREA) + float32 [0,1]
    cv::Mat resized;
    cv::resize(oriented, resized, cv::Size(size, size), 0, 0, cv::INTER_AREA);
    cv::Mat floatMat;
    resized.convertTo(floatMat, CV_32F, 1.0 / 255.0);

    return [NSData dataWithBytes:floatMat.data
                          length:size * size * sizeof(float)];
}

@end
