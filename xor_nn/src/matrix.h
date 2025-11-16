#pragma once
#include <vector>

double dot(const std::vector<double>& a,
           const std::vector<std::vector<double>>& B,
           int col);


std::vector<double> mat_vec_mul(const std::vector<std::vector<double>>& W,
                                const std::vector<double>& x,
                                const std::vector<double>& bias);


std::vector<double> vec_add(const std::vector<double>& a,
                            const std::vector<double>& b);


std::vector<double> vec_scale(const std::vector<double>& v, double s);


std::vector<std::vector<double>> outer(const std::vector<double>& a,
                                       const std::vector<double>& b);


std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>>& M);



std::vector<double> mat_T_vec_mul(const std::vector<std::vector<double>>& W,
                                  const std::vector<double>& v);
