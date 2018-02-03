// interpolate.cpp
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

bool mark_video(std::vector<bool>& marked,
                const std::string& input_path,
                const float display_scale)
{
    std::cout << "Reading input file: " << input_path << "\n";
    cv::VideoCapture cap(input_path);
    if (!cap.isOpened())
    {
        std::cerr << "Failed to open video capture\n";
        return false;
    }

    //count frames
    cap.set(CV_CAP_PROP_POS_AVI_RATIO, 1);
    int frame_count = static_cast<int>(cap.get(CV_CAP_PROP_POS_FRAMES));
    cap.set(CV_CAP_PROP_POS_AVI_RATIO, 0);

    const double fourcc_d = cap.get(CV_CAP_PROP_FOURCC);
    const char* fourcc = reinterpret_cast<const char*>(&fourcc_d);
    std::cout << "FourCC: " << fourcc << "\n";
    std::cout << "Frame count (approx): " << frame_count << "\n";
    int frame_width = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    std::cout << "Frame width: " << frame_width << "\n";
    std::cout << "Frame height: " << frame_height << "\n";
    std::cout << "FPS: " << static_cast<int>(cap.get(CV_CAP_PROP_FPS)) << "\n";
    std::cout << "Frame format: " << static_cast<int>(cap.get(CV_CAP_PROP_FORMAT)) << "\n";
    std::cout << "ISO Speed: " << static_cast<int>(cap.get(CV_CAP_PROP_ISO_SPEED)) << "\n";

    bool backtracked = false;
    marked.resize(frame_count, false);
    bool read_frame = true;
    int fn = 0;
    cv::Mat frame;
    while (fn < frame_count)
    {
        if (backtracked)
        {
            cap.set(CV_CAP_PROP_POS_FRAMES, fn);
            backtracked = false;
        }
        if (read_frame)
        {
            if (!cap.read(frame))
            {
                std::cout << "Frame empty: " << fn << "\n";
                continue;
            }
        }
        std::cout << "Frame number: " << fn << " / " << frame_count << "\n";
        double msec = cap.get(CV_CAP_PROP_POS_MSEC);
        double seconds = std::floor(msec / 1000.0);
        double minutes = std::floor(seconds / 60.0);
        double hours = std::floor(minutes / 60.0);
        minutes -= hours * 60.0;
        seconds -= minutes * 60.0;
        msec -= seconds * 1000.0;
        std::cout << "Time: " << std::setfill('0') << std::setw(2) << static_cast<int>(hours) << ":" << std::setw(2) << static_cast<int>(minutes) << ":" << std::setw(2) << static_cast<int>(seconds) << ":" << std::setw(4) << static_cast<int>(msec) << "\n";
        read_frame = true;
        int fw = frame.cols;
        int fh = frame.rows;

        cv::Size size(static_cast<int>(static_cast<float>(fw) * display_scale), static_cast<int>(static_cast<float>(fh) * display_scale));
        cv::Mat disp;
        cv::resize(frame, disp, size);
        if (marked[fn])
        {
            cv::rectangle(disp, cv::Rect(0, 0, disp.cols, disp.rows), cv::Scalar(0, 0, 255), 5, 8, 0);
        }
        cv::imshow("Display window", disp);

        char key = static_cast<char>(cv::waitKey(0));
        if (key == ' ')
        {
            if (!marked[fn])
            {
                std::cout << "Marked frame " << fn << " for interpolation\n";
                marked[fn] = true;
            }
            else
            {
                std::cout << "Unmarked frame " << fn << "\n";
                marked[fn] = false;
            }
            fn++;
        }
        else if (key == 'b')
        {
            backtracked = true;
            fn--;
        }
        else if (key == 'v')
        {
            backtracked = true;
            fn -= 5;
        }
        else if (key == 'n')
        {
            // next marked frame
            int fnext = fn + 1;
            bool found = false;
            while (fnext < frame_count && !found)
            {
                if (marked[fnext])
                {
                    found = true;
                }
                else
                {
                    fnext++;
                }
            }
            if (found)
            {
                std::cout << "Moving to next marked frame\n";
                backtracked = true;
                fn = fnext;
            }
            else
            {
                read_frame = false;
            }
        }
        else if (key == 'p')
        {
            // previous marked frame
            int fprev = fn - 1;
            bool found = false;
            while (fprev >= 0 && !found)
            {
                if (marked[fprev])
                {
                    found = true;
                }
                else
                {
                    fprev--;
                }
            }
            if (found)
            {
                std::cout << "Moving to previous marked frame\n";
                backtracked = true;
                fn = fprev;
            }
            else
            {
                read_frame = false;
            }
        }
        else if (key == 's')
        {
            std::cout << "Moving to start frame\n";
            backtracked = true;
            fn = 0;
        }
        else if (key == 'q')
        {
            std::cout << "Quitting marking and saving marking file\n";
            cap.release();
            return true;
        }
        else
        {
            fn++;
        }
    }
    cap.release();
    std::cout << "Video marking complete\n";
    return true;
}

bool process_video(cv::VideoWriter& output_video,
                   const std::vector<bool>& marked,
                   const std::string& input_path,
                   const float display_scale)
{
    std::cout << "Reading input file: " << input_path << "\n";
    cv::VideoCapture cap(input_path);
    if (!cap.isOpened())
    {
        std::cerr << "Failed to open video capture\n";
        return false;
    }

    cap.set(CV_CAP_PROP_POS_AVI_RATIO, 1);

    //count frames
    int frame_count = static_cast<int>(cap.get(CV_CAP_PROP_POS_FRAMES));
    cap.set(CV_CAP_PROP_POS_AVI_RATIO, 0);

    const double fourcc_d = cap.get(CV_CAP_PROP_FOURCC);
    const char* fourcc = reinterpret_cast<const char*>(&fourcc_d);
    std::cout << "FourCC: " << fourcc << "\n";
    std::cout << "Frame count (approx): " << frame_count << "\n";
    int frame_width = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    std::cout << "Frame width: " << frame_width << "\n";
    std::cout << "Frame height: " << frame_height << "\n";
    std::cout << "FPS: " << static_cast<int>(cap.get(CV_CAP_PROP_FPS)) << "\n";
    std::cout << "Frame format: " << static_cast<int>(cap.get(CV_CAP_PROP_FORMAT)) << "\n";
    std::cout << "ISO Speed: " << static_cast<int>(cap.get(CV_CAP_PROP_ISO_SPEED)) << "\n";

    int wh = frame_width * frame_height;

    cv::Mat prev_frame;
    int fn = 0;
    while (fn < frame_count)
    {
        cv::Mat frame;
        if (!cap.read(frame))
        {
            std::cout << "Frame empty: "<< fn << "\n";
            continue;
        }
        std::cout << "Frame number: " << fn << " / " << frame_count << "\n";

        if (!marked[fn])
        {
            output_video.write(frame);
            int fw = frame.cols;
            int fh = frame.rows;
            cv::Size size(static_cast<int>(static_cast<float>(fw) * display_scale), static_cast<int>(static_cast<float>(fh) * display_scale));
            cv::Mat disp;
            cv::resize(frame, disp, size);
            if (marked[fn])
            {
                cv::rectangle(disp, cv::Rect(0, 0, disp.cols, disp.rows), cv::Scalar(0, 0, 255), 5, 8, 0);
            }
            cv::imshow("Display window", disp);
            cv::waitKey(30);
            fn++;
        }
        else
        {
            // find next frame
            int search_frame = fn + 1;
            cv::Mat sframe;
            while (marked[search_frame])
            {
                search_frame++;
                while (!cap.read(sframe))
                {
                    std::cout << "Frame empty: " << search_frame << "\n";
                }
            }
            search_frame++;
            while (!cap.read(sframe))
            {
                std::cout << "Frame empty: " << search_frame << "\n";
            }
            search_frame++;
            while (!cap.read(sframe))
            {
                std::cout << "Frame empty: " << search_frame << "\n";
            }
            cv::Mat interp_frame = prev_frame.clone();
            int range = search_frame - fn - 1;
            float inv_range = 1.0f / static_cast<float>(range + 1);
            for (int i = 0; i < range; ++i)
            {
                float alpha = static_cast<float>(i) * inv_range;
#pragma omp parallel for
                for (int t = 0; t < wh; ++t)
                {
                    int x = t % frame_width;
                    int y = t / frame_width;

                    cv::Vec3b prev_texel = prev_frame.at<cv::Vec3b>(y, x);
                    cv::Vec3b s_texel = sframe.at<cv::Vec3b>(y, x);

                    std::array<float, 3> interp_texel;
                    cv::Vec3b itexel;
                    for (int v = 0; v < 3; ++v)
                    {
                        interp_texel[v] = std::max(0.0f, std::min(255.0f, static_cast<float>(prev_texel[v]) * (1.0f - alpha) + static_cast<float>(s_texel[v]) * alpha));
                        itexel[v] = static_cast<char>(interp_texel[v]);
                    }

                    interp_frame.at<cv::Vec3b>(y, x) = itexel;
                }
                std::cout << "interpolating frame: " << fn + i + 1 << "\n";
                output_video.write(interp_frame);

                int fw = frame.cols;
                int fh = frame.rows;
                cv::Size size(static_cast<int>(static_cast<float>(fw) * display_scale), static_cast<int>(static_cast<float>(fh) * display_scale));
                cv::Mat disp;
                cv::resize(interp_frame, disp, size);
                if (marked[fn])
                {
                    cv::rectangle(disp, cv::Rect(0, 0, disp.cols, disp.rows), cv::Scalar(0, 0, 255), 5, 8, 0);
                }
                cv::imshow("Display window", disp);
                cv::waitKey(30);
            }
            fn = search_frame;
        }
        prev_frame = frame.clone();
    }
    cap.release();
    std::cout << "Video processing complete\n";
    return true;
}

int main(int argc, char** argv)
{
    std::cout << "Interpolate\n";
    std::cout << "by Laurence Emms\n";
    std::cout << "Controls:\n";
    std::cout << "Spacebar: Mark frame for interpolation and advance to next frame\n";
    std::cout << "B: Navigate to previous frame\n";
    std::cout << "V: Navigate back 5 frames\n";
    std::cout << "N: Navigate to next marked frame\n";
    std::cout << "P: Navigate to previous marked frame\n";
    std::cout << "S: Navigate to the start frame\n";
    std::cout << "Q: Quit marking and save marked file\n";
    std::cout << "Any other key: Navigate to next frame\n";
    std::cout << "Marked frames are indicated by a red border\n";

    po::options_description desc("Options");
    desc.add_options()
        ("help,h", "Print help message")
        ("version,v", "Print version number")
        ("input,i", po::value<std::string>(), "Input video file")
        ("output,o", po::value<std::string>(), "Output video file")
        ("marked", po::value<std::string>(), "Marked frames file")
        ("force,f", "Force overwriting output")
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
        std::cout << "Interpolate 1.0\n";
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

    if (!fs::exists(input_path.string()))
    {
        std::cerr << "Input file does not exist: " << input_path.string() << "\n";
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

    std::cout << "Reading input file: " << input_path.string() << "\n";
    cv::VideoCapture cap(input_path.string());
    if (!cap.isOpened())
    {
        std::cerr << "Failed to open video capture\n";
        return 1;
    }

    std::cout << "Input format:\n";
    const int fourcc_i = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC));
    const char* fourcc = reinterpret_cast<const char*>(&fourcc_i);
    std::cout << "FourCC: " << fourcc << "\n";
    int frame_width = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    std::cout << "Frame width: " << frame_width << "\n";
    std::cout << "Frame height: " << frame_height << "\n";
    double fps = cap.get(CV_CAP_PROP_FPS);
    std::cout << "FPS: " << fps << "\n";
    double estimated_frame_count = cap.get(CV_CAP_PROP_FRAME_COUNT);
    std::cout << "Estimated frame count: " << estimated_frame_count << "\n";
    std::cout << "Frame format: " << static_cast<int>(cap.get(CV_CAP_PROP_FORMAT)) << "\n";
    std::cout << "ISO Speed: " << static_cast<int>(cap.get(CV_CAP_PROP_ISO_SPEED)) << "\n";

    // count frames
    cap.set(CV_CAP_PROP_POS_AVI_RATIO, 1);
    int frame_count = static_cast<int>(cap.get(CV_CAP_PROP_POS_FRAMES));
    cap.set(CV_CAP_PROP_POS_AVI_RATIO, 0);
    std::cout << "Frame count (approx): " << frame_count << "\n";
    cap.release();

    const float display_scale = 0.4f;
    std::vector<bool> marked(frame_count, false);

    fs::path marked_path;
    if (vm.count("marked") != 0)
    {
        marked_path = vm["marked"].as<std::string>();
        std::cout << "Reading marked data from: " << marked_path.string() << "\n";
        if (fs::exists(marked_path))
        {
            std::ifstream marked_file(marked_path.string().c_str());
            int frame = 0;
            while (marked_file >> frame)
            {
                marked[frame] = true;
            }
        }
    }

    std::cout << "Marking input file: " << input_path.string() << "\n";
    if (!mark_video(marked,
                    input_path.string(),
                    display_scale))
    {
        std::cerr << "Failed to mark video: " << input_path.string() << "\n";
        return 1;
    }

    if (vm.count("marked") != 0)
    {
        std::cout << "Writing marked data to: " << marked_path.string() << "\n";
        std::ofstream marked_file(marked_path.string().c_str());
        for (size_t i = 0; i < marked.size(); ++i)
        {
            if (marked[i])
            {
                marked_file << i << "\n";
            }
        }
    }

    std::cout << "Output file: " << output_path.string() << "\n";

    cv::Size output_size(frame_width, frame_height);
    cv::VideoWriter output_video;
    output_video.open(output_path.string(), fourcc_i, fps, output_size, true);
    if (!output_video.isOpened())
    {
        std::cerr << "Failed to open output video: " << output_path.string() << "\n";
        return 1;
    }

    std::cout << "Processing input file: " << input_path.string() << "\n";
    if (!process_video(output_video,
                       marked,
                       input_path.string(),
                       display_scale))
    {
        std::cerr << "Failed to process video: " << input_path.string() << "\n";
        output_video.release();
        return 1;
    }
    output_video.release();
    std::cout << "Finished writing video: " << output_path.string() << "\n";

    cv::waitKey(0);

    return 0;
}
