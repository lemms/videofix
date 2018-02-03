// mlpclasifier.cpp
// Copyright Laurence Emms 2017

#include "mlpclassifier.h"

namespace classifiers
{
    MLPClassifier::MLPClassifier(bool verbose) : _verbose(verbose), _learning_rate(0.1f), _beta(1.0f), _gen(_rd()), _dist(-1.0, 1.0)
    {
    }

    int MLPClassifier::num_layers() const
    {
        return _layer_counts.size();
    }

    int MLPClassifier::layer_size(int layer) const
    {
        if (layer >= static_cast<int>(_layer_counts.size()))
        {
            std::cerr << "Error: Layer does not exist: " << layer << " / " << _layer_counts.size() << "\n";
            return 0;
        }
        return _layer_counts[layer];
    }

    void MLPClassifier::init(const std::vector<int>& layer_counts, const float learning_rate, const float beta)
    {
        if (layer_counts.size() < 3)
        {
            std::cerr << "Error: MLP has no hidden layers\n";
            return;
        }
        _learning_rate = learning_rate;
        _beta = beta;
        _layer_counts.clear();
        _weights.clear();
        _layers.clear();
        _errors.clear();

        _layer_counts = layer_counts;
        int layers = _layer_counts.size();
        for (int l = 0; l < layers - 1; ++l)
        {
            _weights.push_back(WeightLayer<float, int>(_layer_counts[l], _layer_counts[l + 1]));
            // initialize weights to random values
            const int rows = _weights[l].get_rows();
            const int cols = _weights[l].get_cols();
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    const float value = static_cast<float>(_dist(_gen));
                    _weights[l].set_value(i, j, value);
                }
            }
        }
        // note: the last hidden layer is the output layer
        for (int l = 0; l < layers; ++l)
        {
            _layers.emplace_back(_layer_counts[l], 0.0f);
        }
        for (int l = 1; l < layers; ++l)
        {
            _errors.emplace_back(_layer_counts[l], 0.0f);
        }
    }

    void MLPClassifier::train(const std::vector<float>& input, const std::vector<float>& target)
    {
        feed_forward(input);
        back_propagation(target);
    }

    void MLPClassifier::classify(const std::vector<float>& input, std::vector<float>& output)
    {
        feed_forward(input);
        get_output_layer(output);
    }

    void MLPClassifier::feed_forward(const std::vector<float>& input)
    {
        // process input layer to hidden layer 0
        const int rows = _weights[0].get_rows();
        const int cols = _weights[0].get_cols();
        if (input.size() != rows)
        {
            std::cerr << "Error: Input is the wrong size: " << input.size() << " != " << rows << "\n";
            return;
        }
        _layers[0] = input;

        int layers = static_cast<int>(_layer_counts.size());
        for (int l = 1; l < layers; ++l)
        {
            const int prev_layer_size = _weights[l - 1].get_rows();
            const int current_layer_size = _weights[l - 1].get_cols();
#pragma omp parallel for
            for (int k = 0; k < current_layer_size; ++k)
            {
                _layers[l][k] = 0.0f;
                for (int j = 0; j < prev_layer_size; ++j)
                {
                    _layers[l][k] += _weights[l - 1].get_value(j, k) * _layers[l - 1][j];
                }
                _layers[l][k] = 1.0f / (1.0f + std::exp(-_beta * _layers[l][k]));
            }
        }
    }

    void MLPClassifier::back_propagation(const std::vector<float>& target)
    {
        if (target.size() != _layer_counts.back())
        {
            std::cerr << "Target layer is not the correct size: " << target.size() << " != " << _layer_counts.back() << "\n";
            return;
        }
        int layers = static_cast<int>(_layer_counts.size());
        {
            int l = layers - 1;
            if (_verbose)
                std::cout << "compute error layer " << l << "\n";
            const int current_layer_size = _layers[l].size();
            if (_verbose)
                std::cout << " current layer size: " << current_layer_size << "\n";
#pragma omp parallel for
            for (int k = 0; k < current_layer_size; ++k) // current_layer_size layer
            {
                float value = _layers[l][k]; // cuurrent layer value
                _errors[l - 1][k] = (target[k] - value) * value * (1.0f - value);
            }
        }
        for (int l = layers - 2; l > 0; --l)
        {
            if (_verbose)
                std::cout << "compute error layer " << l << " -> " << l + 1 << "\n";
            int current_layer_size = _layers[l].size();
            int next_layer_size = _layers[l + 1].size();
#pragma omp parallel for
            for (int j = 0; j < current_layer_size; ++j) // current layer
            {
                float activation = _layers[l][j]; // current layer activation
                _errors[l - 1][j] = 0.0f;
                for (int k = 0; k < next_layer_size; ++k) // next layer
                {
                    _errors[l - 1][j] += _weights[l].get_value(j, k) * _errors[l][k];
                }
                _errors[l - 1][j] *= activation * (1.0f - activation);
            }
        }

        for (int l = layers - 2; l >= 0; --l)
        {
            if (_verbose)
                std::cout << "updating weights " << l + 1 << "\n";
            int current_layer_size = _layers[l].size();
            int next_layer_size = _layers[l + 1].size();
#pragma omp parallel for
            for (int j = 0; j < current_layer_size; ++j) // current layer
            {
                float activation = _layers[l][j];
                for (int k = 0; k < next_layer_size; ++k) // next layer
                {
                    _weights[l].set_value(j, k, _weights[l].get_value(j, k) + _learning_rate * _errors[l][k] * activation);
                }
            }
        }
    }

    void MLPClassifier::get_output_layer(std::vector<float>& output)
    {
        output.clear();
        if (output.size() != _layers.back().size())
        {
            output.resize(_layers.back().size());
        }
        std::copy(_layers.back().begin(), _layers.back().end(), output.begin());
    }

    void MLPClassifier::write(std::ofstream& stream) const
    {
        stream << "nn\n";
        int layers = static_cast<int>(_layer_counts.size());
        stream << layers << "\n";
        stream << _learning_rate << "\n";
        stream << _beta << "\n";
        for (int l = 0; l < layers; ++l)
        {
            stream << _layer_counts[l] << "\n";
        }

        for (int l = 0; l < layers - 1; ++l)
        {
            const int rows = _weights[l].get_rows();
            const int cols = _weights[l].get_cols();
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    stream << _weights[l].get_value(i, j) << " ";
                }
            }
            stream << "\n";
        }
    }

    void MLPClassifier::read(std::ifstream& stream)
    {
        std::string type;
        stream >> type;
        if (type != "nn")
        {
            std::cerr << "Error: MLPClassifier is not a neural network.\n";
            return;
        }

        int layers = 0;
        stream >> layers;
        if (layers < 3)
        {
            std::cerr << "Error: MLP has no hidden layers\n";
            return;
        }
        stream >> _learning_rate;
        stream >> _beta;

        _layer_counts.clear();
        _weights.clear();
        _layers.clear();
        _errors.clear();

        for (int l = 0; l < layers; ++l)
        {
            int layer = 0;
            stream >> layer;
            _layer_counts.push_back(layer);
        }

        for (int l = 0; l < layers - 1; ++l)
        {
            _weights.push_back(WeightLayer<float, int>(_layer_counts[l], _layer_counts[l + 1]));
        }

        for (int l = 0; l < layers - 1; ++l)
        {
            const int rows = _weights[l].get_rows();
            const int cols = _weights[l].get_cols();
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    float value = 0.0f;
                    stream >> value;
                    _weights[l].set_value(i, j, value);
                }
            }
        }
        for (int l = 0; l < layers; ++l)
        {
            _layers.emplace_back(_layer_counts[l], 0.0f);
        }
        for (int l = 1; l < layers; ++l)
        {
            _errors.emplace_back(_layer_counts[l], 0.0f);
        }
    }
}
