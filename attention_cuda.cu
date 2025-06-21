#include <stdio.h>

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

// Helper for ceiling division
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

// CUDA error checking helper
#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t err = call;                                                                    \
        if (err != cudaSuccess) {                                                                  \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": "           \
                      << cudaGetErrorString(err) << std::endl;                                     \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

// CUDA kernel for matrix multiplication (C = A * B)
__global__ void matmul_kernel(int m, int n, int k, const float* A, const float* B, float* C) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

// CUDA wrapper for matrix multiplication
void matmul(int m, int n, int k, const float* A, const float* B, float* C) {
    // Define grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim(CEIL_DIV(n, blockDim.x), CEIL_DIV(m, blockDim.y));

    matmul_kernel<<<gridDim, blockDim>>>(m, n, k, A, B, C);

    CUDA_CHECK(cudaGetLastError());
}

__global__ void softmax(int M, int N, const float* A, float* B) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M) {
        // Find the maximum value in this row for numerical stability
        float max_val = A[row * N];
        for (int col = 1; col < N; ++col) {
            max_val = fmaxf(max_val, A[row * N + col]);
        }

        // Compute exp(x - max) for each element and sum them
        float sum = 0.0f;
        for (int col = 0; col < N; ++col) {
            // Subtract max_val for numerical stability
            float exp_val = expf(A[row * N + col] - max_val);
            B[row * N + col] = exp_val; // Store temporarily
            sum += exp_val;
        }

        // Normalize by dividing by the sum
        for (int col = 0; col < N; ++col) {
            B[row * N + col] /= sum;
        }
    }
}

__global__ void scale(int M, int N, const float* A, float* B, float scale) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < M && y < N) {
        B[x * N + y] = A[x * N + y] * scale;
    }
}

__global__ void mask(int M, int N, const float* A, float* B) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < M && y < N) {
        B[x * N + y] = y <= x ? A[x * N + y] : -INFINITY;
    }
}
__global__ void transpose(int M, int N, const float* A, float* B) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < M && y < N) {
        B[y * M + x] = A[x * N + y];
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

    constexpr int X = 16;
    dim3 gridDim(CEIL_DIV(m, X), CEIL_DIV(n, X));
    // 32 * 32 = 1024 thread per block
    dim3 blockDim(X, X, 1);
    int threadsPerBlock = 256;

    transpose<<<gridDim, blockDim>>>(n, dk, k, scratch_n_dk);                // dk x n
    matmul(m, n, dk, q, scratch_n_dk, scratch_m_n_2); // m x n
    std::swap(scratch_m_n_1, scratch_m_n_2);
    scale<<<gridDim, blockDim>>>(
        m,
        n,
        scratch_m_n_1,
        scratch_m_n_2,
        1.f / std::sqrt(float(dk))); // m x n
    std::swap(scratch_m_n_1, scratch_m_n_2);
    if (causal) {
        mask<<<gridDim, blockDim>>>(m, n, scratch_m_n_1, scratch_m_n_2); // m x n
        std::swap(scratch_m_n_1, scratch_m_n_2);
    }
    softmax<<<CEIL_DIV(m, threadsPerBlock), threadsPerBlock>>>(
        m,
        n,
        scratch_m_n_1,
        scratch_m_n_2); // m x n
    std::swap(scratch_m_n_1, scratch_m_n_2);
    matmul(m, dv, n, scratch_m_n_1, v, a); // m x dv

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        // Handle error...
    }
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

// Measure attention runtime (in milliseconds)
double measure_attention(int m, int n, int dk, int dv, bool causal, int iters = 5) {
    reseed(1ULL * m * dk * dv * iters);

    double time_ms = 0.0;

    // Allocate and initialize host memory
    float* h_q = new float[m * dk];
    float* h_k = new float[n * dk];
    float* h_v = new float[n * dv];
    float* h_a = new float[m * dv];
    randomize(m * dk, h_q);
    randomize(n * dk, h_k);
    randomize(n * dv, h_v);

    // Allocate device memory
    float *d_q, *d_k, *d_v, *d_a, *d_scratch_n_dk, *d_scratch_m_n_1, *d_scratch_m_n_2;
    CUDA_CHECK(cudaMalloc(&d_q, m * dk * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k, n * dk * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v, n * dv * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_a, m * dv * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scratch_n_dk, n * dk * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scratch_m_n_1, m * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scratch_m_n_2, m * n * sizeof(float)));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_q, h_q, m * dk * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k, n * dk * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v, n * dv * sizeof(float), cudaMemcpyHostToDevice));

    // Warm-up run
    attention(
        m,
        n,
        dk,
        dv,
        d_q,
        d_k,
        d_v,
        d_a,
        d_scratch_n_dk,
        d_scratch_m_n_1,
        d_scratch_m_n_2,
        causal);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < iters; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        attention(
            m,
            n,
            dk,
            dv,
            d_q,
            d_k,
            d_v,
            d_a,
            d_scratch_n_dk,
            d_scratch_m_n_1,
            d_scratch_m_n_2,
            causal);
        CUDA_CHECK(cudaDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        time_ms += duration.count() / 1000.0;
    }

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_a, d_a, m * dv * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << m << ' ' << dv << '\n' << std::fixed << std::setprecision(6);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < dv; ++j) {
            std::cout << h_a[i * dv + j] << " \n"[j == dv - 1];
        }
    }

    // Free memory
    delete[] h_q;
    delete[] h_k;
    delete[] h_v;
    delete[] h_a;

    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_scratch_n_dk));
    CUDA_CHECK(cudaFree(d_scratch_m_n_1));
    CUDA_CHECK(cudaFree(d_scratch_m_n_2));

    return time_ms / iters;
}

// CUDA kernel for copying head-specific data
__global__ void copy_head_data_kernel(
    int size,
    int head_size,
    int num_heads,
    int head,
    const float* src,
    float* dst) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        const int row = idx / head_size;
        const int col = idx % head_size;
        dst[row * head_size + col] = src[row * (num_heads * head_size) + head * head_size + col];
    }
}

// Multi-Head Attention implementation for CUDA
void multi_head_attention(
    int m,
    int n,
    int d_model,
    int h,
    int dk,
    int dv,
    const float* q,       // m x d_model (device pointer)
    const float* k,       // n x d_model (device pointer)
    const float* v,       // n x d_model (device pointer)
    const float* Wq,      // d_model x d_model (device pointer)
    const float* Wk,      // d_model x d_model (device pointer)
    const float* Wv,      // d_model x d_model (device pointer)
    const float* Wo,      // (h*dv) x d_model (device pointer)
    float* out,           // m x d_model (device pointer)
    float* Q_all,         // m x (h*dk) (device pointer)
    float* K_all,         // n x (h*dk) (device pointer)
    float* V_all,         // n x (h*dv) (device pointer)
    float* head_out,      // m x (h*dv) (device pointer)
    float* scratch_n_dk,  // n x dk (device pointer)
    float* scratch_m_n_1, // m x n (device pointer)
    float* scratch_m_n_2, // m x n (device pointer)
    float* Qh_buf,        // m * dk (device pointer)
    float* Kh_buf,        // n * dk (device pointer)
    float* Vh_buf,        // n * dv (device pointer)
    bool causal) {

    // 1. Project inputs into Q, K, V spaces (one matmul per projection)
    matmul(m, h * dk, d_model, q, Wq, Q_all);
    matmul(n, h * dk, d_model, k, Wk, K_all);
    matmul(n, h * dv, d_model, v, Wv, V_all);

    // 2. Compute attention for each head independently
    for (int head = 0; head < h; head++) {
        // Configure grid and block dimensions for copy kernels
        dim3 q_block_dim(256);
        dim3 q_grid_dim(CEIL_DIV(m * dk, q_block_dim.x));

        dim3 k_block_dim(256);
        dim3 k_grid_dim(CEIL_DIV(n * dk, k_block_dim.x));

        dim3 v_block_dim(256);
        dim3 v_grid_dim(CEIL_DIV(n * dv, v_block_dim.x));

        // Copy Q, K, V data for this head
        /*
        copy_head_data_kernel<<<q_grid_dim, q_block_dim>>>(m * dk, dk, h, head, Q_all, Qh_buf);

        copy_head_data_kernel<<<k_grid_dim, k_block_dim>>>(n * dk, dk, h, head, K_all, Kh_buf);

        copy_head_data_kernel<<<v_grid_dim, v_block_dim>>>(n * dv, dv, h, head, V_all, Vh_buf);
        */

        // copy Q_all[:, head*dk:(head+1)*dk] into Qh_buf
        for (int i = 0; i < m; ++i) {
            cudaMemcpy(Qh_buf + i * dk, Q_all + i * (h * dk) + head * dk, dk * sizeof(float), cudaMemcpyDeviceToDevice);
        }
        // same for K
        for (int i = 0; i < n; ++i) {
            cudaMemcpy(Kh_buf + i * dk, K_all + i * (h * dk) + head * dk, dk * sizeof(float), cudaMemcpyDeviceToDevice);
        }
        // and for V
        for (int i = 0; i < n; ++i) {
            cudaMemcpy(Vh_buf + i * dv, V_all + i * (h * dv) + head * dv, dv * sizeof(float), cudaMemcpyDeviceToDevice);
        }

        // Wait for copy operations to complete
        CUDA_CHECK(cudaGetLastError());

        // Single-head attention
        attention(
            m,
            n,
            dk,
            dv,
            Qh_buf,
            Kh_buf,
            Vh_buf,
            head_out + head * m * dv, // Pointer arithmetic for output position
            scratch_n_dk,
            scratch_m_n_1,
            scratch_m_n_2,
            causal);
    }

    // 3. Project concatenated head outputs to model dimension
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

    // Allocate and initialize host memory
    float* h_q = new float[m * d_model];
    float* h_k = new float[n * d_model];
    float* h_v = new float[n * d_model];
    float* h_Wq = new float[d_model * d_model];
    float* h_Wk = new float[d_model * d_model];
    float* h_Wv = new float[d_model * d_model];
    float* h_Wo = new float[(h * dv) * d_model];
    float* h_out = new float[m * d_model];

    randomize(m * d_model, h_q);
    randomize(n * d_model, h_k);
    randomize(n * d_model, h_v);
    randomize(d_model * d_model, h_Wq);
    randomize(d_model * d_model, h_Wk);
    randomize(d_model * d_model, h_Wv);
    randomize((h * dv) * d_model, h_Wo);

    // Allocate device memory
    float *d_q, *d_k, *d_v, *d_Wq, *d_Wk, *d_Wv, *d_Wo, *d_out;
    float *d_Q_all, *d_K_all, *d_V_all, *d_head_out;
    float *d_scratch_n_dk, *d_scratch_m_n_1, *d_scratch_m_n_2;
    float *d_Qh_buf, *d_Kh_buf, *d_Vh_buf;

    CUDA_CHECK(cudaMalloc(&d_q, m * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k, n * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v, n * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Wq, d_model * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Wk, d_model * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Wv, d_model * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Wo, (h * dv) * d_model * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, m * d_model * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_Q_all, m * h * dk * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K_all, n * h * dk * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V_all, n * h * dv * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_head_out, m * h * dv * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_scratch_n_dk, n * dk * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scratch_m_n_1, m * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scratch_m_n_2, m * n * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_Qh_buf, m * dk * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Kh_buf, n * dk * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Vh_buf, n * dv * sizeof(float)));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_q, h_q, m * d_model * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k, n * d_model * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v, n * d_model * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Wq, h_Wq, d_model * d_model * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Wk, h_Wk, d_model * d_model * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Wv, h_Wv, d_model * d_model * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Wo, h_Wo, (h * dv) * d_model * sizeof(float), cudaMemcpyHostToDevice));

    // Warm-up run
    multi_head_attention(
        m,
        n,
        d_model,
        h,
        dk,
        dv,
        d_q,
        d_k,
        d_v,
        d_Wq,
        d_Wk,
        d_Wv,
        d_Wo,
        d_out,
        d_Q_all,
        d_K_all,
        d_V_all,
        d_head_out,
        d_scratch_n_dk,
        d_scratch_m_n_1,
        d_scratch_m_n_2,
        d_Qh_buf,
        d_Kh_buf,
        d_Vh_buf,
        causal);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Measure execution time
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
            d_q,
            d_k,
            d_v,
            d_Wq,
            d_Wk,
            d_Wv,
            d_Wo,
            d_out,
            d_Q_all,
            d_K_all,
            d_V_all,
            d_head_out,
            d_scratch_n_dk,
            d_scratch_m_n_1,
            d_scratch_m_n_2,
            d_Qh_buf,
            d_Kh_buf,
            d_Vh_buf,
            causal);
        CUDA_CHECK(cudaDeviceSynchronize());

        auto end = std::chrono::high_resolution_clock::now();
        time_ms +=
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    }

    // Copy results back to host (optional, just to verify)
    CUDA_CHECK(cudaMemcpy(h_out, d_out, m * d_model * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << m << ' ' << d_model << '\n' << std::fixed << std::setprecision(6);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < d_model; ++j) {
            std::cout << h_out[i * d_model + j] << " \n"[j == d_model - 1];
        }
    }

    // Clean up
    delete[] h_q;
    delete[] h_k;
    delete[] h_v;
    delete[] h_Wq;
    delete[] h_Wk;
    delete[] h_Wv;
    delete[] h_Wo;
    delete[] h_out;

    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_Wq));
    CUDA_CHECK(cudaFree(d_Wk));
    CUDA_CHECK(cudaFree(d_Wv));
    CUDA_CHECK(cudaFree(d_Wo));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_Q_all));
    CUDA_CHECK(cudaFree(d_K_all));
    CUDA_CHECK(cudaFree(d_V_all));
    CUDA_CHECK(cudaFree(d_head_out));
    CUDA_CHECK(cudaFree(d_scratch_n_dk));
    CUDA_CHECK(cudaFree(d_scratch_m_n_1));
    CUDA_CHECK(cudaFree(d_scratch_m_n_2));
    CUDA_CHECK(cudaFree(d_Qh_buf));
    CUDA_CHECK(cudaFree(d_Kh_buf));
    CUDA_CHECK(cudaFree(d_Vh_buf));

    return time_ms / iters;
}

int main() {
    // Configure CUDA device
    cudaSetDevice(0);

    // (m,n): sequence lengths
    // d_model: model dimension
    // h:       # heads
    // dk, dv:  per‐head key/value dims
    // causal:  mask flag
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
                  << "\n"
                  << std::string(56, '-') << "\n";
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

#undef CEIL_DIV
