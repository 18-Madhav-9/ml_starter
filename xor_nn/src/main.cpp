#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

#include "neural_network.h"
#include "../data/xor_data.h"

static void print_prediction(const std::vector<double>& input,
                             double prediction,
                             double expected)
{
    std::cout << "  Input: ["
              << static_cast<int>(input[0]) << ", "
              << static_cast<int>(input[1]) << "]"
              << "  →  Prediction: " << std::fixed << std::setprecision(4)
              << prediction
              << "  (rounded: " << (prediction >= 0.5 ? 1 : 0) << ")"
              << "  Expected: " << static_cast<int>(expected)
              << "\n";
}

int main()
{
    const int    EPOCHS       = 10000;
    const int    LOG_EVERY    = 1000;   
    const double LEARNING_RATE = 0.5;

    
    NeuralNetwork nn(2, 2, 1, LEARNING_RATE);

    std::vector<Sample> data = get_xor_data();

    std::cout << "========================================\n";
    std::cout << "  XOR Neural Network — C++ from scratch\n";
    std::cout << "========================================\n";
    std::cout << "Architecture : 2 → 2 → 1\n";
    std::cout << "Activation   : Sigmoid\n";
    std::cout << "Loss         : MSE\n";
    std::cout << "Learning rate: " << LEARNING_RATE << "\n";
    std::cout << "Epochs       : " << EPOCHS << "\n\n";
    std::cout << "--- Training ---\n";
    for (int epoch = 1; epoch <= EPOCHS; ++epoch) {

        double total_loss = 0.0;

        
        for (const auto& sample : data) {
            nn.train(sample.input, sample.output);

            
            auto out = nn.predict(sample.input);
            total_loss += nn.mse(out, sample.output);
        }

        if (epoch % LOG_EVERY == 0 || epoch == 1) {
            double avg_loss = total_loss / data.size();
            std::cout << "Epoch " << std::setw(6) << epoch
                      << "  |  Avg MSE: " << std::fixed << std::setprecision(6)
                      << avg_loss << "\n";
        }
    }

    std::cout << "\n--- Final Predictions ---\n";
    bool all_correct = true;
    for (const auto& sample : data) {
        auto out      = nn.predict(sample.input);
        double pred   = out[0];
        double expected = sample.output[0];
        print_prediction(sample.input, pred, expected);
        if ((pred >= 0.5 ? 1 : 0) != static_cast<int>(expected))
            all_correct = false;
    }

    std::cout << "\nResult: " << (all_correct ? "✅ All XOR outputs correct!"
                                              : "❌ Some outputs are wrong.")
              << "\n";
    std::cout << "========================================\n";

    return 0;
}
