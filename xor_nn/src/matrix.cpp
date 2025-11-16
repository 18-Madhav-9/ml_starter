#include "matrix.h"
#include <stdexcept>

double dot(const std::vector<double>& a,
           const std::vector<std::vector<double>>& B,
           int col)
{
    double result = 0.0;
    for (size_t k = 0; k < a.size(); ++k)
        result += a[k] * B[k][col];
    return result;
}

std::vector<double> mat_vec_mul(const std::vector<std::vector<double>>& W,
                                const std::vector<double>& x,
                                const std::vector<double>& bias)
{
    size_t rows = W.size();
    std::vector<double> result(rows, 0.0);
    for (size_t i = 0; i < rows; ++i) {
        result[i] = bias[i];
        for (size_t j = 0; j < x.size(); ++j)
            result[i] += W[i][j] * x[j];
    }
    return result;
}

std::vector<double> vec_add(const std::vector<double>& a,
                            const std::vector<double>& b)
{
    std::vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i)
        result[i] = a[i] + b[i];
    return result;
}

std::vector<double> vec_scale(const std::vector<double>& v, double s)
{
    std::vector<double> result(v.size());
    for (size_t i = 0; i < v.size(); ++i)
        result[i] = v[i] * s;
    return result;
}

std::vector<std::vector<double>> outer(const std::vector<double>& a,
                                       const std::vector<double>& b)
{
    std::vector<std::vector<double>> result(a.size(),
                                            std::vector<double>(b.size(), 0.0));
    for (size_t i = 0; i < a.size(); ++i)
        for (size_t j = 0; j < b.size(); ++j)
            result[i][j] = a[i] * b[j];
    return result;
}

std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>>& M)
{
    if (M.empty()) return {};
    size_t rows = M.size(), cols = M[0].size();
    std::vector<std::vector<double>> T(cols, std::vector<double>(rows, 0.0));
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            T[j][i] = M[i][j];
    return T;
}


std::vector<double> mat_T_vec_mul(const std::vector<std::vector<double>>& W,
                                  const std::vector<double>& v)
{
    size_t rows = W.size();
    size_t cols = W[0].size();
    std::vector<double> result(cols, 0.0);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            result[j] += W[i][j] * v[i];
    return result;
}
