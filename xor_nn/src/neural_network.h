#pragma once
#include <vector>

class NeuralNetwork {
public:
    
    NeuralNetwork(int input_size, int hidden_size, int output_size,
                  double learning_rate = 0.5);

    
    std::vector<double> predict(const std::vector<double>& input);

    
    void train(const std::vector<double>& input,
               const std::vector<double>& target);

    
    double mse(const std::vector<double>& output,
               const std::vector<double>& target) const;

private:
    int n_input, n_hidden, n_output;
    double lr;  

    
    std::vector<std::vector<double>> W1;  
    std::vector<double>              B1;  

    
    std::vector<std::vector<double>> W2;  
    std::vector<double>              B2;  

    
    std::vector<double> z1;   
    std::vector<double> a1;   
    std::vector<double> z2;   
    std::vector<double> a2;   

    
    double sigmoid(double x) const;
    double sigmoid_derivative(double x) const;  

    
    void init_weights();
};
