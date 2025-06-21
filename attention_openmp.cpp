#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <omp.h>

void matmul(int M, int N, int K, const float* A, const float* B, float* C);
void softmax(int M, int N, const float* A, float* B);
void scale(int M, int N, const float* A, float* B, float scale);
void mask(int M, int N, const float* A, float* B);
void transpose(int M, int N, const float* A, float* B);

// Matrix multiplication: C[M,N] = A[M,K] * B[K,N]
void matmul(int M, int N, int K, const float* A, const float* B, float* C) {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            auto sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Row-wise softmax: B[i,j] = exp(A[i,j]) / sum_j(exp(A[i,j]))
void softmax(int M, int N, const float* A, float* B) {
#pragma omp parallel for
    for (int i = 0; i < M; i++) {
        // Find max value in this row for numerical stability
        float max_val = -FLT_MAX;
        for (int j = 0; j < N; j++) {
            max_val = std::max(max_val, A[i * N + j]);
        }

        // Compute exp(x - max) for each element and sum
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            float exp_val = std::exp(A[i * N + j] - max_val);
            B[i * N + j] = exp_val;
            sum += exp_val;
        }

        // Normalize by sum
        for (int j = 0; j < N; j++) {
            B[i * N + j] /= sum;
        }
    }
}

// Element-wise scaling: B[i,j] = A[i,j] * scale
void scale(int M, int N, const float* A, float* B, float scale) {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            B[i * N + j] = A[i * N + j] * scale;
        }
    }
}

// Causal triangular mask: B[i,j] = A[i,j] if j <= i, -infinity otherwise
void mask(int M, int N, const float* A, float* B) {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (j <= i) {
                B[i * N + j] = A[i * N + j];
            } else {
                B[i * N + j] = -std::numeric_limits<float>::infinity(); // Mask future positions
            }
        }
    }
}

// Matrix transpose: B[N,M] = A[M,N]^T
void transpose(int M, int N, const float* A, float* B) {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            B[j * M + i] = A[i * N + j];
        }
    }
}

void attention(
    int m,
    int n,
    int dk,
    int dv,
    const float* q,       // m x dk
    const float* k,       // n x dk
    const float* v,       // n x dv
    float* a,             // m x dv
    float* scratch_n_dk,  // n x dk
    float* scratch_m_n_1, // m x n
    float* scratch_m_n_2, // m x n
    bool causal) {
    transpose(n, dk, k, scratch_n_dk);                // dk x n
    matmul(m, n, dk, q, scratch_n_dk, scratch_m_n_2); // m x n
    std::swap(scratch_m_n_1, scratch_m_n_2);
    scale(m, n, scratch_m_n_1, scratch_m_n_2, 1.f / std::sqrt(float(dk))); // m x n
    std::swap(scratch_m_n_1, scratch_m_n_2);
    if (causal) {
        mask(m, n, scratch_m_n_1, scratch_m_n_2); // m x n
        std::swap(scratch_m_n_1, scratch_m_n_2);
    }
    softmax(m, n, scratch_m_n_1, scratch_m_n_2); // m x n
    std::swap(scratch_m_n_1, scratch_m_n_2);
    matmul(m, dv, n, scratch_m_n_1, v, a); // m x dv
}

std::mt19937 gen(0);

void reseed(uint64_t seed) {
    gen = std::mt19937(seed);
}

void randomize(int n, float* a, float mean = 0.0f, float stddev = 1.0f) {
    std::normal_distribution<float> dist(mean, stddev);
    for (int i = 0; i < n; ++i) {
        a[i] = dist(gen);
    }
}

double measure_attention(int m, int n, int dk, int dv, bool causal, int iters = 5) {
    reseed(1ULL * m * dk * dv * iters);

    double time_ms = 0.0;

    float* q = new float[m * dk];
    randomize(m * dk, q);
    float* k = new float[n * dk];
    randomize(n * dk, k);
    float* v = new float[n * dv];
    randomize(n * dv, v);
    float* a = new float[m * dv];
    float* scratch_n_dk = new float[n * dk];
    float* scratch_m_n_1 = new float[m * n];
    float* scratch_m_n_2 = new float[m * n];

    for (int i = 0; i < iters; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        attention(m, n, dk, dv, q, k, v, a, scratch_n_dk, scratch_m_n_1, scratch_m_n_2, causal);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        time_ms += duration.count() / 1000.0;
    }

    std::cout << m << ' ' << dv << '\n' << std::fixed << std::setprecision(6);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < dv; ++j) {
            std::cout << a[i * dv + j] << " \n"[j == dv - 1];
        }
    }

    delete[] q, delete[] k, delete[] v, delete[] a, delete[] scratch_m_n_1, delete[] scratch_m_n_2,
        delete[] scratch_n_dk;
    return time_ms / iters;
}

// Multi-Head Attention (MHA)
//  - m: target sequence length
//  - n: source sequence length
//  - d_model: model dimension (must equal h * dk)
//  - h: number of heads
//  - dk: per-head key/query dimension
//  - dv: per-head value dimension
//  - q, k, v: input projections of shape m/n x d_model
//  - Wq, Wk, Wv: weight matrices for Q/K/V of shape d_model x d_model
//  - Wo: output projection of shape (h*dv) x d_model
//  - out: result buffer of shape m x d_model
//  - Q_all, K_all, V_all, head_out: pre-allocated scratch buffers
//  - scratch_n_dk: scratch for transpose, size n x dk
//  - scratch_m_n_1, scratch_m_n_2: score scratch, size m x n

void multi_head_attention(
    int m,
    int n,
    int d_model,
    int h,
    int dk,
    int dv,
    const float* q,  // m x d_model
    const float* k,  // n x d_model
    const float* v,  // n x d_model
    const float* Wq, // d_model x d_model
    const float* Wk, // d_model x d_model
    const float* Wv, // d_model x d_model
    const float* Wo, // (h*dv) x d_model
    float* out,      // m x d_model
    float* Q_all,    // m x (h*dk)
    float* K_all,    // n x (h*dk)
    float* V_all,    // n x (h*dv)
    float* head_out, // m x (h*dv)

    float* scratch_n_dk,  // n x dk
    float* scratch_m_n_1, // m x n
    float* scratch_m_n_2, // m x n

    float* Qh_buf, // m * dk
    float* Kh_buf, // n * dk
    float* Vh_buf, // n * dv
    bool causal) {
    // 1. Project inputs into Q, K, V spaces (one matmul per projection)
    //    Q_all: m x (h*dk)
    matmul(m, h * dk, d_model, q, Wq, Q_all);
    //    K_all: n x (h*dk)
    matmul(n, h * dk, d_model, k, Wk, K_all);
    //    V_all: n x (h*dv)
    matmul(n, h * dv, d_model, v, Wv, V_all);

    // 2. Compute attention for each head independently
    for (int head = 0; head < h; head++) {
        // copy Q_all[:, head*dk:(head+1)*dk] into Qh_buf
        for (int i = 0; i < m; ++i) {
            std::memcpy(Qh_buf + i * dk, Q_all + i * (h * dk) + head * dk, dk * sizeof(float));
        }
        // same for K
        for (int i = 0; i < n; ++i) {
            std::memcpy(Kh_buf + i * dk, K_all + i * (h * dk) + head * dk, dk * sizeof(float));
        }
        // and for V
        for (int i = 0; i < n; ++i) {
            std::memcpy(Vh_buf + i * dv, V_all + i * (h * dv) + head * dv, dv * sizeof(float));
        }

        // Single-head attention (no causal masking by default)
        attention(
            m,
            n,
            dk,
            dv,
            Qh_buf,
            Kh_buf,
            Vh_buf,
            head_out + head * dv,
            scratch_n_dk,
            scratch_m_n_1,
            scratch_m_n_2,
            causal);
    }

    // 3. Concatenate all head outputs (already in head_out buffer) and project
    //    head_out: m x (h*dv)  ->  out: m x d_model
    matmul(m, d_model, h * dv, head_out, Wo, out);
}

// Measure multi-head attention runtime (in milliseconds)
double measure_multi_head_attention(
    int m,
    int n,
    int d_model,
    int h,
    int dk,
    int dv,
    bool causal,
    int iters = 5) {
    reseed(1ULL * m * n * d_model * h * iters);

    // Allocate and randomize inputs and weights
    float* q = new float[m * d_model];
    randomize(m * d_model, q);
    float* k = new float[n * d_model];
    randomize(n * d_model, k);
    float* v = new float[n * d_model];
    randomize(n * d_model, v);
    float* Wq = new float[d_model * d_model];
    randomize(d_model * d_model, Wq);
    float* Wk = new float[d_model * d_model];
    randomize(d_model * d_model, Wk);
    float* Wv = new float[d_model * d_model];
    randomize(d_model * d_model, Wv);
    float* Wo = new float[(h * dv) * d_model];
    randomize((h * dv) * d_model, Wo);
    float* out = new float[m * d_model];

    // Scratch buffers
    float* Q_all = new float[m * h * dk];
    float* K_all = new float[n * h * dk];
    float* V_all = new float[n * h * dv];
    float* head_out = new float[m * h * dv];
    float* scratch_n_dk = new float[n * dk];
    float* scratch_m_n_1 = new float[m * n];
    float* scratch_m_n_2 = new float[m * n];
    float* Qh_buf = new float[m * dk];
    float* Kh_buf = new float[n * dk];
    float* Vh_buf = new float[n * dv];

    double time_ms = 0.0;
    for (int i = 0; i < iters; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        multi_head_attention(
            m,
            n,
            d_model,
            h,
            dk,
            dv,
            q,
            k,
            v,
            Wq,
            Wk,
            Wv,
            Wo,
            out,
            Q_all,
            K_all,
            V_all,
            head_out,
            scratch_n_dk,
            scratch_m_n_1,
            scratch_m_n_2,
            Qh_buf,
            Kh_buf,
            Vh_buf,
            causal);
        auto end = std::chrono::high_resolution_clock::now();
        time_ms +=
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    }

    std::cout << m << ' ' << d_model << '\n' << std::fixed << std::setprecision(6);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < d_model; ++j) {
            std::cout << out[i * d_model + j] << " \n"[j == d_model - 1];
        }
    }


    // Clean up
    delete[] q;
    delete[] k;
    delete[] v;
    delete[] Wq;
    delete[] Wk;
    delete[] Wv;
    delete[] Wo;
    delete[] out;
    delete[] Q_all;
    delete[] K_all;
    delete[] V_all;
    delete[] head_out;
    delete[] scratch_n_dk;
    delete[] scratch_m_n_1;
    delete[] scratch_m_n_2;
    delete[] Qh_buf;
    delete[] Kh_buf;
    delete[] Vh_buf;

    return time_ms / iters;
}

int main() {
    // (m,n): sequence lengths
    // d_model: model dimension
    // h:       # heads
    // dk, dv:  per‐head key/value dims
    // causal:  mask flag
    //
    // We pick moderate sizes so the sequential version still finishes quickly:
    //  - shorter:  m=n=128, d_model=128, h=4 → dk=dv=32
    //  - longer:   m=n=512, d_model=256, h=8 → dk=dv=32

    struct Config {
        int m, n, d_model, h, dk, dv;
        bool causal;
    };
    std::vector<Config> configs = {
        // m, n, d_model, h, dk, dv, causal, time_ms
        { 256,  256,  256,  8, 64, 64, false},
        { 256,  256,  256,  8, 64, 64,  true},
        { 512,  512,  512,  8, 64, 64, false},
        { 512,  512,  512,  8, 64, 64,  true},
        {1024, 1024,  512, 16, 64, 64, false},
        {1024, 1024,  512, 16, 64, 64,  true},
        {2048, 2048, 1024, 16, 64, 64, false},
        {2048, 2048, 1024, 16, 64, 64,  true},
        {4096, 2048, 1024, 16, 64, 64, false},
        {4096, 2048, 1024, 16, 64, 64,  true},
    };

    const int iters = 5;
    constexpr bool multi_head = false;

    if constexpr (multi_head) {
        std::clog << std::setw(6) << "m" << std::setw(6) << "n" << std::setw(8) << "d_model"
                  << std::setw(4) << "h" << std::setw(6) << "dk" << std::setw(6) << "dv"
                  << std::setw(8) << "causal" << std::setw(12) << "Time(ms)"
                  << "\n";
    } else {
        std::clog << std::setw(6) << "m" << std::setw(6) << "n" << std::setw(6) << "dk"
                  << std::setw(6) << "dv" << std::setw(8) << "causal" << std::setw(12) << "Time(ms)"
                  << "\n"
                  << std::string(56, '-') << "\n";
    }

    for (auto& c : configs) {
        double t = 0.0;

        if constexpr (multi_head) {
            t = measure_multi_head_attention(c.m, c.n, c.d_model, c.h, c.dk, c.dv, c.causal, iters);
        } else {
            t = measure_attention(c.m, c.n, c.dk, c.dv, c.causal, iters);
        }
        if constexpr (multi_head) {
            std::clog << std::setw(6) << c.m << std::setw(6) << c.n << std::setw(8) << c.d_model
                      << std::setw(4) << c.h << std::setw(6) << c.dk << std::setw(6) << c.dv
                      << std::setw(8) << (c.causal ? "yes" : "no") << std::setw(12) << std::fixed
                      << std::setprecision(2) << t << "\n";
        } else {
            std::clog << std::setw(6) << c.m << std::setw(6) << c.n << std::setw(6) << c.dk
                      << std::setw(6) << c.dv << std::setw(8) << (c.causal ? "yes" : "no")
                      << std::setw(12) << std::fixed << std::setprecision(2) << t << "\n";
        }
    }

    std::cout << "-1 -1\n" << std::flush;

    return 0;
}
