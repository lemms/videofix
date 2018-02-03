// classify.cpp
// Copyright Laurence Emms 2017

#include <cmath>
#include <iostream>
#include <array>
#include <numeric>
#include <algorithm>
#include <list>
#include <random>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <opencv2/opencv.hpp>

#include <mlpclassifier.h>

namespace po = boost::program_options;
namespace fs = boost::filesystem;

template <typename Classifier>
bool classify(Classifier& classifier,
              std::vector<bool>& marked,
              const std::string& input_path,
              const int w,
              const int h,
              const int f,
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

    std::list<cv::Mat> prev_frames;
    for (int fn = 0; fn < frame_count; ++fn)
    {
        cv::Mat frame;
        if (!cap.read(frame))
        {
            std::cout << "Frame empty: "<< fn << "\n";
            continue;
        }
        prev_frames.push_front(frame.clone());
        if (fn == 0)
        {
            // preload f frames
            for (int i = 0; i < f - 1; ++i)
            {
                prev_frames.push_front(frame.clone());
            }
        }
        while (static_cast<int>(prev_frames.size()) > f)
        {
            prev_frames.pop_back();
        }
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
        int w_offset = w / 2;
        int h_offset = h / 2;
        int fw = frame.cols;
        int fh = frame.rows;

        float mean_output = 0.0f;
        int output_count = 0;
        std::vector<float> output_vector;
        std::vector<float> input_vector(w * f * h + 1);
        input_vector[w * h * f] = -1.0f; // bias node
        int stride = w;
        int total = 0;
        for (int y = h_offset; y + h_offset < fh; y += stride)
        {
            for (int x = w_offset; x + w_offset < fw; x += stride)
            {
                std::list<cv::Mat>::iterator current_frame = prev_frames.begin();

                int it_f = 0;
                while (current_frame != prev_frames.end())
                {
                    int it_y = 0;
                    for (int y_o = -h_offset; y_o < h_offset; ++y_o)
                    {
                        int it_x = 0;
                        for (int x_o = -w_offset; x_o < w_offset; ++x_o)
                        {
                            cv::Vec3b texel = current_frame->at<cv::Vec3b>(y + y_o, x + x_o);
                            float luminance = (0.2126f * static_cast<float>(texel[0]) + 0.7512f * static_cast<float>(texel[1]) + 0.0722f * static_cast<float>(texel[2])) / 255.0f;
                            input_vector[it_x + it_y * w + it_f * w * h] = luminance;
                            it_x++;
                        }
                        it_y++;
                    }
                    current_frame++;
                    it_f++;
                }

                classifier.classify(input_vector, output_vector);
                if (output_vector.empty())
                {
                    std::cerr << "\n";
                    continue;
                }
                mean_output += output_vector[0];
                if (output_vector[0] > 0.5f)
                {
                    output_count++;
                }
                total++;
            }
        }

        float output_fraction = static_cast<float>(output_count) / static_cast<float>(total);

        if (verbose)
        {
            std::cout << static_cast<int>(output_fraction * 100.0f) << "% of frames classified as marked\n";
            std::cout << "Mean output: " << mean_output / static_cast<float>(total) << "\n";
        }
        float marked_threshold = 0.0f;
        if (output_fraction > marked_threshold)
        {
            marked[fn] = true;
        }
        if (show)
        {
            cv::Size size(static_cast<int>(static_cast<float>(fw) * display_scale), static_cast<int>(static_cast<float>(fh) * display_scale));
            cv::Mat disp;
            cv::resize(frame, disp, size);
            if (marked[fn])
            {
                cv::rectangle(disp, cv::Rect(0, 0, disp.cols, disp.rows), cv::Scalar(0, 0, 255), 5, 8, 0);
            }
            cv::imshow("Display window", disp);
            cv::waitKey(15);
        }
        if (verbose)
        {
            std::cout << "Frame classified\n";
        }
    }
    cap.release();
    std::cout << "Training complete\n";
    return true;
}

int main(int argc, char** argv)
{
    std::srand(static_cast<unsigned int>(std::time(0)));
    std::cout << "Classify\n";
    std::cout << "by Laurence Emms\n";

    po::options_description desc("Options");
    desc.add_options()
        ("help,h", "Print help message")
        ("version,v", "Print version number")
        ("input,i", po::value<std::string>(), "Input video file")
        ("classifier,c", po::value<std::string>(), "Classifier file")
        ("marked", po::value<std::string>(), "Marked frames file")
        ("show,s", "Display output")
        ("verbose", "Force verbose output")
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
        std::cout << "Train 1.0\n";
        return 0;
    }

    if (vm.count("input") == 0)
    {
        std::cerr << "Input file not specified\n";
        return 1;
    }

    if (vm.count("classifier") == 0)
    {
        std::cerr << "Classifier file not specified\n";
        return 1;
    }

    if (vm.count("marked") == 0)
    {
        std::cerr << "Marked file not specified\n";
        return 1;
    }


    fs::path input_path(vm["input"].as<std::string>());
    fs::path classifier_path(vm["classifier"].as<std::string>());
    fs::path marked_path(vm["marked"].as<std::string>());
    
    if (!fs::exists(input_path.string()))
    {
        std::cerr << "Input file does not exist: " << input_path.string() << "\n";
        return 1;
    }

    std::cout << "Reading input file: " << input_path.string() << "\n";
    cv::VideoCapture cap(input_path.string());
    if (!cap.isOpened())
    {
        std::cerr << "Failed to open video capture\n";
        return 1;
    }

    int w = 8;
    int h = 8;
    int f = 4;

    classifiers::MLPClassifier classifier;
    if (fs::exists(classifier_path.string()))
    {
        std::cout << "Reading classifier file: " << classifier_path.string() << "\n";
        std::ifstream classifier_file(classifier_path.string().c_str());
        classifier.read(classifier_file);
        int layers = classifier.num_layers();
        if (layers <= 0)
        {
            std::cerr << "Error: Classifier has no layers\n";
            return -1;
        }
        std::cout << "Read classifier with " << layers << " layers\n";
        for (int l = 0; l < layers; ++l)
        {
            std::cout << l << ": " << classifier.layer_size(l) << "\n";
        }
    }
    else
    {
        std::vector<int> layer_sizes;
        layer_sizes.push_back(w * h * f + 1);
        layer_sizes.push_back(w * h * f + 1);
        layer_sizes.push_back(w * h * f + 1);
        layer_sizes.push_back(1);
        classifier.init(layer_sizes);
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

    bool verbose = vm.count("verbose") != 0;
    bool show = vm.count("show") != 0;
    if (show)
    {
        cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
    }

    float display_scale = 0.4f;
    std::vector<bool> marked(frame_count, false);
    std::cout << "Classifying input file: " << input_path.string() << "\n";
    if (!classify(classifier,
                  marked,
                  input_path.string(),
                  w,
                  h,
                  f,
                  display_scale,
                  show,
                  verbose))
    {
        std::cerr << "Failed to classify on video: " << input_path.string() << "\n";
        return 1;
    }

    std::cout << "Writing marked data to: " << marked_path.string() << "\n";
    std::ofstream marked_file(marked_path.string().c_str());
    for (size_t f = 0; f < marked.size(); ++f)
    {
        if (marked[f])
        {
            marked_file << f << "\n";
        }
    }

    std::cout << "Finished classifying video: " << input_path.string() << "\n";

    cv::waitKey(0);

    return 0;
}
