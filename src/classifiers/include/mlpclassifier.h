// mlpclassifier.h
// Copyright Laurence Emms 2017

#ifndef MLP_CLASSIFIER
#define MLP_CLASSIFIER

#include <cmath>
#include <string>
#include <iostream>
#include <numeric>
#include <vector>
#include <random>
#include <fstream>

namespace classifiers
{
    template <typename T, typename S = size_t>
    class WeightLayer
    {
    public:
        WeightLayer(const S rows, const S cols);
        void set_value(const S i, const S j, const T value);
        T get_value(const S i, const S j) const;
        S get_rows() const;
        S get_cols() const;
    private:
        std::vector<T> _weights;
        const S _rows;
        const S _cols;
    };

    template <typename T, typename S>
    WeightLayer<T, S>::WeightLayer(const S rows, const S cols) : _rows(rows), _cols(cols)
    {
        _weights.resize(_rows * _cols, static_cast<T>(0));
    }

    template <typename T, typename S>
    void WeightLayer<T, S>::set_value(const S i, const S j, const T value)
    {
        if (i < 0 || j < 0 || i >= _rows || j >= _cols)
        {
            std::cerr << "Error: Out of bounds set value " << i << ", " << j << "\n";
            return;
        }
        _weights[j + i * _cols] = value;
    }

    template <typename T, typename S>
    T WeightLayer<T, S>::get_value(const S i, const S j) const
    {
        if (i < 0 || j < 0 || i >= _rows || j >= _cols)
        {
            std::cerr << "Error: Out of bounds get value " << i << ", " << j << "\n";
            return 0.0f;
        }
        return _weights[j + i * _cols];
    }

    template <typename T, typename S>
    S WeightLayer<T, S>::get_rows() const
    {
        return _rows;
    }

    template <typename T, typename S>
    S WeightLayer<T, S>::get_cols() const
    {
        return _cols;
    }

    // multi-layer perceptron classifier
    class MLPClassifier
    {
    public:
        MLPClassifier(bool verbose = false);
        int num_layers() const;
        int layer_size(int layer) const;
        void init(const std::vector<int>& layer_counts, const float learning_rate = 0.1f, const float beta = 1.0f);
        void train(const std::vector<float>& input, const std::vector<float>& target);
        void classify(const std::vector<float>& input, std::vector<float>& output);
        void feed_forward(const std::vector<float>& input);
        void back_propagation(const std::vector<float>& target);
        void get_output_layer(std::vector<float>& output);
        void write(std::ofstream& stream) const;
        void read(std::ifstream& stream);
    private:
        bool _verbose;
        float _learning_rate;
        float _beta;
        std::vector<int> _layer_counts;
        std::vector<WeightLayer<float, int>> _weights;
        std::vector<std::vector<float>> _layers;
        std::vector<std::vector<float>> _errors;
        std::random_device _rd;
        std::mt19937 _gen;
        std::uniform_real_distribution<> _dist;
    };
}

#endif // MLP_CLASSIFIER
