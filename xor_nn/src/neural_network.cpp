#include "neural_network.h"
#include "matrix.h"

#include <cmath>
#include <cstdlib>   
#include <ctime>     
#include <stdexcept>

NeuralNetwork::NeuralNetwork(int input_size, int hidden_size, int output_size,
                             double learning_rate)
    : n_input(input_size),
      n_hidden(hidden_size),
      n_output(output_size),
      lr(learning_rate)
{
    
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    init_weights();
}


void NeuralNetwork::init_weights()
{
    auto rand_val = []() -> double {
        return (static_cast<double>(std::rand()) / RAND_MAX) - 0.5;
    };

    
    W1.assign(n_hidden, std::vector<double>(n_input));
    for (auto& row : W1)
        for (auto& w : row)
            w = rand_val();

    
    B1.assign(n_hidden, 0.0);

    
    W2.assign(n_output, std::vector<double>(n_hidden));
    for (auto& row : W2)
        for (auto& w : row)
            w = rand_val();

    
    B2.assign(n_output, 0.0);
}

double NeuralNetwork::sigmoid(double x) const
{
    return 1.0 / (1.0 + std::exp(-x));
}

double NeuralNetwork::sigmoid_derivative(double x) const
{
    double s = sigmoid(x);
    return s * (1.0 - s);
}

std::vector<double> NeuralNetwork::predict(const std::vector<double>& input)
{
    
    z1 = mat_vec_mul(W1, input, B1);
    a1.resize(n_hidden);
    for (int i = 0; i < n_hidden; ++i)
        a1[i] = sigmoid(z1[i]);

    
    z2 = mat_vec_mul(W2, a1, B2);
    a2.resize(n_output);
    for (int i = 0; i < n_output; ++i)
        a2[i] = sigmoid(z2[i]);

    return a2;
}


double NeuralNetwork::mse(const std::vector<double>& output,
                          const std::vector<double>& target) const
{
    double loss = 0.0;
    for (size_t i = 0; i < output.size(); ++i) {
        double diff = output[i] - target[i];
        loss += diff * diff;
    }
    return loss / output.size();
}


void NeuralNetwork::train(const std::vector<double>& input,
                          const std::vector<double>& target)
{
    
    predict(input);

    
    std::vector<double> delta2(n_output);
    for (int i = 0; i < n_output; ++i)
        delta2[i] = (a2[i] - target[i]) * sigmoid_derivative(z2[i]);

    
    
    std::vector<double> delta1(n_hidden);
    std::vector<double> back = mat_T_vec_mul(W2, delta2);  
    for (int j = 0; j < n_hidden; ++j)
        delta1[j] = back[j] * sigmoid_derivative(z1[j]);

    
    auto dW2 = outer(delta2, a1);         
    for (int i = 0; i < n_output; ++i) {
        for (int j = 0; j < n_hidden; ++j)
            W2[i][j] -= lr * dW2[i][j];
        B2[i] -= lr * delta2[i];
    }

    
    auto dW1 = outer(delta1, input);      
    for (int i = 0; i < n_hidden; ++i) {
        for (int j = 0; j < n_input; ++j)
            W1[i][j] -= lr * dW1[i][j];
        B1[i] -= lr * delta1[i];
    }
}
