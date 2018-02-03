// undistort.cpp
// Copyright Laurence Emms 2017

#include <cmath>
#include <iostream>
#include <array>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <opencv2/opencv.hpp>

namespace po = boost::program_options;
namespace fs = boost::filesystem;

void rotate(cv::Mat& mat, int angle)
{
    if (angle == 90)
    {
        cv::transpose(mat, mat);
        cv::flip(mat, mat, 1);
    }
    else if (angle == 180)
    {
        cv::flip(mat, mat, -1);
    }
    else if (angle == 270)
    {
        cv::transpose(mat, mat);
        cv::flip(mat, mat, 0);
    }
}

bool process_video(cv::VideoWriter& output_video,
                   const cv::Size& output_size,
                   const std::string& input_path,
                   const int output_width,
                   const int output_height,
                   const float aspect_ratio,
                   const int rotation_angle,
                   const float xscale,
                   const float yscale,
                   const float gain,
                   const float bias,
                   const float gamma,
                   const float alpha,
                   const float display_scale,
                   const bool show,
                   const bool verbose)
{
    cv::VideoCapture cap(input_path);
    if (!cap.isOpened())
    {
        std::cerr << "Failed to open video capture\n";
        return false;
    }

    const double fourcc_d = cap.get(CV_CAP_PROP_FOURCC);
    const char* fourcc = reinterpret_cast<const char*>(&fourcc_d);
    std::cout << "FourCC: " << fourcc << "\n";
    int frame_width = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    std::cout << "Frame width: " << frame_width << "\n";
    std::cout << "Frame height: " << frame_height << "\n";
    std::cout << "FPS: " << static_cast<int>(cap.get(CV_CAP_PROP_FPS)) << "\n";
    std::cout << "Frame format: " << static_cast<int>(cap.get(CV_CAP_PROP_FORMAT)) << "\n";
    std::cout << "ISO Speed: " << static_cast<int>(cap.get(CV_CAP_PROP_ISO_SPEED)) << "\n";

    // count frames
    cap.set(CV_CAP_PROP_POS_AVI_RATIO, 1);
    int frame_count = static_cast<int>(cap.get(CV_CAP_PROP_POS_FRAMES));
    cap.set(CV_CAP_PROP_POS_AVI_RATIO, 0);
    std::cout << "Frame count (approx): " << frame_count << "\n";

    for (int fn = 0; fn < frame_count; ++fn)
    {
        if (verbose)
        {
            std::cout << "Frame number: " << fn << " / " << frame_count << "\n";
            double msec = cap.get(CV_CAP_PROP_POS_MSEC);
            double seconds = std::floor(msec / 1000.0);
            double minutes = std::floor(seconds / 60.0);
            double hours = std::floor(minutes / 60.0);
            minutes -= hours * 60.0;
            seconds -= minutes * 60.0;
            msec -= seconds * 1000.0;
            std::cout << "Time: " << std::setfill('0') << std::setw(2) << static_cast<int>(hours) << ":" << std::setw(2) << static_cast<int>(minutes) << ":" << std::setw(2) << static_cast<int>(seconds) << ":" << std::setw(4) << static_cast<int>(msec) << "\n";
        }
        cv::Mat frame;
        if (!cap.read(frame))
        {
            std::cout << "Frame empty: "<< fn << "\n";
            continue;
        }
        int fw = frame.cols;
        int fh = frame.rows;
        rotate(frame, rotation_angle);
        if (rotation_angle == 90 || rotation_angle == 270)
        {
            fw = fh;
            fh = fw;
        }
        fw = static_cast<int>(static_cast<float>(fw) * xscale);
        fh = static_cast<int>(static_cast<float>(fh) * yscale);
        cv::Size scale_size(fw, fh);
        cv::Mat scale_frame;
        cv::resize(frame, scale_frame, scale_size);

        cv::Mat undistorted_frame = scale_frame.clone();

        float ffw = static_cast<float>(fw);
        float ffh = static_cast<float>(fh);
        float f_aspect = ffw / ffh;
        float half_f_aspect = f_aspect * 0.5f;
        float inv_f_aspect = 1.0f / f_aspect;
        int wh = fw * fh;
        int hw = fw / 2;
        int hh = fh / 2;
#pragma omp parallel for
        for (int t = 0; t < wh; ++t)
        {
            int x = t % fw;
            int y = t / fw;
            // convert to normalized coordinates
            float nx = static_cast<float>(x) / ffw * f_aspect;
            float ny = static_cast<float>(y) / ffh;
            // convert to centered coordinates
            float dx = nx - half_f_aspect;
            float dy = ny - 0.5f;
            // compute distorted radius
            float rd = std::sqrt(dx * dx + dy * dy);
            // compute undistorted radius
            float ru = rd / (1.0f - alpha * rd * rd);
            float nux = dx * ru / rd;
            float nuy = dy * ru / rd;
            // convert to texel coordinates
            float ux = (nux + half_f_aspect) * ffw * inv_f_aspect;
            float uy = (nuy + 0.5f) * ffh;
            // offsets for linear interpolation
            float ox = ux - std::floor(ux);
            float oy = uy - std::floor(uy);
            int sx = static_cast<int>(ux);
            int sy = static_cast<int>(uy);
            if (sx >= -1 && sy >= -1 && sx < fw && sy < fh)
            {
                // clamp edges on interpolation
                int sxfloor = std::max(0, sx);
                int syfloor = std::max(0, sy);
                int sxplus = std::min(sx + 1, fw - 1);
                int syplus = std::min(sy + 1, fh - 1);
                cv::Vec3b t00 = scale_frame.at<cv::Vec3b>(syfloor, sxfloor);
                cv::Vec3b t01 = scale_frame.at<cv::Vec3b>(syfloor, sxplus);
                cv::Vec3b t10 = scale_frame.at<cv::Vec3b>(syplus, sxfloor);
                cv::Vec3b t11 = scale_frame.at<cv::Vec3b>(syplus, sxplus);
                std::array<float, 3> t0;
                std::array<float, 3> t1;
                std::array<float, 3> tx;
                cv::Vec3b texel;
                for (int i = 0; i < 3; ++i)
                {
                    t0[i] = static_cast<float>(t00[i]) / 255.0f * (1.0f - ox) + static_cast<float>(t01[i]) / 255.0f * ox;
                    t1[i] = static_cast<float>(t10[i]) / 255.0f * (1.0f - ox) + static_cast<float>(t11[i]) / 255.0f * ox;
                    tx[i] = t0[i] * (1.0f - oy) + t1[i] * oy;
                    texel[i] = static_cast<char>(std::max(0.0f, std::min(255.0f, (std::pow(tx[i], gamma) * gain + bias) * 255.0f)));
                }
                undistorted_frame.at<cv::Vec3b>(y, x) = texel;
            }
            else
            {
                undistorted_frame.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
            }
        }

        int width_offset = 0;
        int height_offset = 0;
        int crop_width = fw;
        int crop_height = fh;
        if (output_width > output_height)
        {
            crop_height = static_cast<int>(static_cast<float>(crop_width) / aspect_ratio + 0.5f);
            height_offset = (fh - crop_height) / 2;
        }
        else
        {
            crop_width = static_cast<int>(static_cast<float>(crop_height) * aspect_ratio + 0.5f);
            width_offset = (fw - crop_width) / 2;
        }
        cv::Rect roi(width_offset, height_offset, crop_width, crop_height);
        if (verbose)
        {
            std::cout << "cw: " << crop_width << " ch: " << crop_height << "\n";
            std::cout << "a: " << static_cast<float>(crop_width) / static_cast<float>(crop_height) << "\n";
        }

        cv::Mat cropped_frame = undistorted_frame(roi);
        // sharpen image
        cv::Mat image;
        cv::GaussianBlur(cropped_frame, image, cv::Size(0, 0), 3);
        cv::addWeighted(cropped_frame, 1.5, image, -0.5, 0, image);

        cv::Mat write_frame;
        cv::resize(image, write_frame, output_size);
        output_video.write(write_frame);
        
        if (show)
        {
            cv::Size size(static_cast<int>(static_cast<float>(crop_width) * display_scale), static_cast<int>(static_cast<float>(crop_height) * display_scale));
            cv::Mat disp;
            cv::resize(image, disp, size);
            cv::imshow("Display window", disp);
            cv::waitKey(15);
        }
    }
    cap.release();
    std::cout << "Video processing complete\n";
    return true;
}

int main(int argc, char** argv)
{
    std::cout << "Undistort\n";
    std::cout << "by Laurence Emms\n";
    po::options_description desc("Options");
    int target_width = 0;
    int target_height = 0;
    int rotation_angle = 0;
    float target_alpha = 0.0f;
    float xscale = 1.0f;
    float yscale = 1.0f;
    float target_gain = 1.0f;
    float target_bias = 0.0f;
    float target_gamma = 1.0f;
    desc.add_options()
        ("help,h", "Print help message")
        ("version,v", "Print version number")
        ("input,i", po::value<std::string>(), "Input video file")
        ("output,o", po::value<std::string>(), "Output video file")
        ("show,s", "Display output")
        ("force,f", "Force overwriting output")
        ("rotation,r", po::value<int>(&rotation_angle), "Rotate image by this angle (CW in degrees)")
        ("verbose", "Force verbose output")
        ("width", po::value<int>(&target_width), "Output width")
        ("height", po::value<int>(&target_height), "Output height")
        ("alpha", po::value<float>(&target_alpha), "Distortion correction")
        ("xscale,x", po::value<float>(&xscale), "X Scale")
        ("yscale,y", po::value<float>(&yscale), "Y Scale")
        ("gain", po::value<float>(&target_gain), "Gain")
        ("bias", po::value<float>(&target_bias), "Bias")
        ("gamma", po::value<float>(&target_gamma), "Gamma")
        ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help"))
    {
        std::cout << desc << "\n";
        return 0;
    }

    if (vm.count("version"))
    {
        std::cout << "Undistort 1.0\n";
        return 0;
    }

    if (vm.count("input") == 0)
    {
        std::cerr << "Input file not specified\n";
        return 1;
    }

    if (vm.count("output") == 0)
    {
        std::cerr << "Output file not specified\n";
        return 1;
    }

    fs::path input_path(vm["input"].as<std::string>());
    fs::path output_path(vm["output"].as<std::string>());

    if (input_path.string() == output_path.string())
    {
        std::cerr << "Input file must be different from output file.\n";
        return 1;
    }

    if (fs::exists(output_path))
    {
        std::cerr << "Output file already exists\n";
        if (vm.count("force") == 0)
        {
            return 1;
        }
    }

    if (vm.count("rotation") == 0)
    {
        rotation_angle = vm["rotation"].as<int>();
    }
    if (rotation_angle != 0 && rotation_angle != 90 && rotation_angle != 180 && rotation_angle != 270)
    {
        std::cerr << "Invalid rotation angle. The only valid angles are 0, 90, 180, and 270.\n";
        return 1;
    }

    std::vector<std::string> input_paths;
    boost::split(input_paths, input_path.string(), boost::is_any_of(","));
    if (input_paths.empty())
    {
        std::cerr << "No input path supplied.\n";
        return 1;
    }
    std::vector<std::string> trimmed_paths;
    for (size_t i = 0; i < input_paths.size(); ++i)
    {
        std::string path = input_paths[i];
        boost::algorithm::trim(path);
        if (!fs::exists(path))
        {
            std::cerr << "Input file does not exist: " << path << "\n";
            return 1;
        }
        trimmed_paths.push_back(path);
    }

    cv::VideoCapture cap(trimmed_paths[0]);
    if (!cap.isOpened())
    {
        std::cerr << "Failed to open video capture\n";
        return 1;
    }
    cap.set(CV_CAP_PROP_POS_AVI_RATIO, 1);

    //count frames
    int frame_count = static_cast<int>(cap.get(CV_CAP_PROP_POS_FRAMES));
    cap.set(CV_CAP_PROP_POS_AVI_RATIO, 0);

    std::cout << "Input format:\n";
    const int fourcc_i = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC));
    const char* fourcc = reinterpret_cast<const char*>(&fourcc_i);
    std::cout << "FourCC: " << fourcc << "\n";
    std::cout << "Frame count (approx): " << frame_count << "\n";
    int frame_width = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    std::cout << "Frame width: " << frame_width << "\n";
    std::cout << "Frame height: " << frame_height << "\n";
    double fps = cap.get(CV_CAP_PROP_FPS);
    std::cout << "FPS: " << fps << "\n";
    std::cout << "Frame format: " << static_cast<int>(cap.get(CV_CAP_PROP_FORMAT)) << "\n";
    std::cout << "ISO Speed: " << static_cast<int>(cap.get(CV_CAP_PROP_ISO_SPEED)) << "\n";
    cap.release();

    int output_width = frame_width;
    int output_height = frame_height;
    if (rotation_angle == 90 || rotation_angle == 270)
    {
        output_width = frame_height;
        output_height = frame_width;
    }
    if (vm.count("width") != 0)
    {
        output_width = target_width;
    }
    if (vm.count("height") != 0)
    {
        output_height = target_height;
    }

    std::cout << "Output file: " << output_path.string() << "\n";
    std::cout << "Output dimensions: " << output_width << " x " << output_height << "\n";

    float aspect_ratio = static_cast<float>(output_width) / static_cast<float>(output_height);
    std::cout << "Aspect ratio: " << aspect_ratio << "\n";
    std::cout << "Rotation angle: " << rotation_angle << "\n";
    std::cout << "xy scale: " << xscale << ", " << yscale << "\n";

    cv::Size output_size(output_width, output_height);
    cv::VideoWriter output_video;
    output_video.open(output_path.string(), fourcc_i, fps, output_size, true);
    if (!output_video.isOpened())
    {
        std::cerr << "Failed to open output video: " << output_path.string() << "\n";
        return 1;
    }

    bool show = vm.count("show") != 0;
    if (show)
    {
        cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
    }

    float gain = 1.0f;
    if (vm.count("gain") != 0)
    {
        gain = target_gain;
    }
    float bias = 0.0f;
    if (vm.count("bias") != 0)
    {
        bias = target_bias;
    }
    float gamma = 1.0f;
    if (vm.count("gamma") != 0)
    {
        gamma = target_gamma;
    }
    float alpha = -1.0f;
    if (vm.count("alpha") != 0)
    {
        alpha = target_alpha;
    }

    bool verbose = vm.count("verbose") != 0;

    const float display_scale = 0.4f;
    for (size_t i = 0; i < trimmed_paths.size(); ++i)
    {
        std::cout << "Processing input file " << i << ": " << trimmed_paths[i] << "\n";
        if (!process_video(output_video,
                           output_size,
                           trimmed_paths[i],
                           output_width,
                           output_height,
                           aspect_ratio,
                           rotation_angle,
                           xscale,
                           yscale,
                           gain,
                           bias,
                           gamma,
                           alpha,
                           display_scale,
                           show,
                           verbose))
        {
            std::cerr << "Failed to process input file: " << i << "\n";
            output_video.release();
            return 1;
        }
    }
    output_video.release();
    std::cout << "Finished writing video: " << output_path.string() << "\n";

    cv::waitKey(0);

    return 0;
}
