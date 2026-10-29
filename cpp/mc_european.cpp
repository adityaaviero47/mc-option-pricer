// Monte Carlo European call under GBM with antithetic variates.
// Build: g++ -O3 -march=native -o mc_european cpp/mc_european.cpp
// Usage: ./mc_european S K T r sigma n_paths seed
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

double bs_call(double S, double K, double T, double r, double sig) {
    double d1 = (std::log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * std::sqrt(T));
    double d2 = d1 - sig * std::sqrt(T);
    auto N = [](double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); };
    return S * N(d1) - K * std::exp(-r * T) * N(d2);
}

int main(int argc, char** argv) {
    double S = 100, K = 100, T = 1, r = 0.05, sig = 0.2;
    long n = 10000000; unsigned seed = 42;
    if (argc >= 8) { S = atof(argv[1]); K = atof(argv[2]); T = atof(argv[3]); r = atof(argv[4]);
                     sig = atof(argv[5]); n = atol(argv[6]); seed = atoi(argv[7]); }
    auto t0 = std::chrono::high_resolution_clock::now();
    std::mt19937_64 gen(seed);
    std::normal_distribution<double> N01(0.0, 1.0);
    const double drift = (r - 0.5 * sig * sig) * T, vol = sig * std::sqrt(T), disc = std::exp(-r * T);
    double sum = 0, sum2 = 0;
    const long pairs = n / 2;
    for (long i = 0; i < pairs; ++i) {
        double z = N01(gen);
        double p1 = std::fmax(S * std::exp(drift + vol * z) - K, 0.0);
        double p2 = std::fmax(S * std::exp(drift - vol * z) - K, 0.0);
        double v = 0.5 * (p1 + p2) * disc;
        sum += v; sum2 += v * v;
    }
    double mean = sum / pairs, se = std::sqrt((sum2 / pairs - mean * mean) / pairs);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
    double exact = bs_call(S, K, T, r, sig);
    std::printf("{\"price\": %.6f, \"se\": %.6f, \"bs\": %.6f, \"abs_err\": %.6f, \"paths\": %ld, \"ms\": %.1f}\n",
                mean, se, exact, std::fabs(mean - exact), n, ms);
    return 0;
}
