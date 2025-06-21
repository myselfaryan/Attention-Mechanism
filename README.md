---

# 🚀 Accelerating Scaled Dot-Product Attention using OpenMP and CUDA

![Project Type](https://img.shields.io/badge/Project-HPC_Accelerated-blueviolet?style=for-the-badge&logo=nvidia)

## 📌 Overview

This project explores the **parallelization of the Scaled Dot-Product Attention mechanism**—a core component of modern Transformer architectures—using **OpenMP** (for multi-core CPUs) and **CUDA** (for NVIDIA GPUs). We compare both parallel implementations against a sequential C++ baseline to analyze performance benefits across various configurations.

> ⚙️ Developed as part of the **High Performance Computing (HPC)** course at our institution.

---


## 🔍 What is Attention Mechanism?

The **Attention Mechanism** allows models to focus on relevant parts of the input when producing output, especially critical in natural language processing and sequence modeling.

### 📐 Scaled Dot-Product Attention
Given matrices **Q** (Query), **K** (Key), and **V** (Value), attention is computed as:
```

Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V

```
This operation is at the heart of Transformers and scales **quadratically with the input sequence length**, making it a computational bottleneck for large inputs.

---

## 🎯 Project Goals

- ✅ Implement a **correct and efficient sequential C++ baseline** for:
  - Self-Attention
  - Cross-Attention
  - Multi-Head Attention

- ⚡ Develop parallelized versions using:
  - **OpenMP** for CPU-based parallelism
  - **CUDA** for GPU acceleration

- 📊 Benchmark and analyze performance improvements
- 🔬 Explore the impact of parameters like:
  - Sequence length
  - Embedding dimension
  - Number of heads

---

## 🔧 Technical Stack

| Component | Technology |
|----------|-------------|
| Language | C++ (C++23 with optimization flags) |
| CPU Parallelism | OpenMP v4 |
| GPU Acceleration | CUDA 12.8 |
| Verification | Python (via `verify.py`) |
| Hardware | Intel Core i5 (12th Gen), NVIDIA RTX 3050 |

---

## 🧠 Attention Variants Explored

| Variant | Description |
|--------|-------------|
| **Self-Attention** | Sequence attends to itself (Q, K, V from same source) |
| **Cross-Attention** | One sequence attends to another |
| **Multi-Head Attention** | Runs multiple attention layers in parallel to capture diverse features |

---

## 📈 Performance Benchmarks

### ✅ Sequential C++ Baseline (Sample)

| Seq Length | Emb Dim | Heads | Self-Attn (ms) | Cross-Attn (ms) | Multi-Head (ms) |
|------------|---------|-------|----------------|------------------|-----------------|
| 256        | 128     | 8     | 11.24          | 11.12            | 97.69           |
| 2048       | 512     | 32    | 2292.63        | 2340.37          | 76268.52        |

> 🚨 Multi-Head attention dominates compute cost, especially at higher dimensions.

---

## 🧵 OpenMP Implementation Details

- Used `#pragma omp parallel for` to parallelize:
  - Matrix multiplication (Q × Kᵀ)
  - Scaling, softmax computation
- Ensured thread-safety with local accumulators
- Tuned `OMP_NUM_THREADS` for optimal throughput
- Observed ideal performance scaling up to 8 threads

---

## ⚙️ CUDA Implementation Highlights

- Modular CUDA kernels:
  - `matmul()`, `softmax()`, `scale()`, `mask()`, `transpose()`
- Key Optimizations:
  - Used `__shared__` memory for caching
  - Ensured coalesced memory accesses
  - Minimized CPU-GPU sync overhead with `cudaDeviceSynchronize`

---

## ✅ Verification Strategy

To ensure correctness across implementations:

- Compared OpenMP/CUDA outputs against sequential C++
- Verified shape match and numerical closeness (error < `1e-4`)
- Automated using `verify.py`

---

## 📁 Project Structure

```

📦 Attention-Mechanism-HPC
├── attention_cuda.cu        # CUDA kernel implementations
├── attention_openmp.cpp     # CPU-parallelized OpenMP versions
├── attention_sequential.cpp # Baseline C++ implementations
├── Makefile               
├── README.md         
├── EndReview.pdf           
├── MidReview.pdf
├── time.log
└── verify.py                # Validation script

```

---

## 📚 References

- 📄 [Attention is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al.
- 🎥 [3Blue1Brown - Attention Mechanism Explained](https://www.youtube.com/watch?v=K0vD8c2p0wM)
- 🎓 [Andrej Karpathy - GPT from Scratch](https://karpathy.ai/)

---

## ✨ Team

- **Saurabh Pal (S20220010196)**
- **Sundar R    (S20220010215)**
- **Aryan Sharma(S20220010021)**

---

### 🙏 Acknowledgements

A special thank you to our **course instructor, Dr. Bheemappa Halavar**, for his continuous support, expert guidance, and insightful feedback throughout the project.  
> Without his mentorship, this work would not have been possible.

📅 **Final Submission Date:** May 10th, 2025

---

## 🧠 Key Takeaways

- Parallelization **dramatically reduces execution time**, especially for large inputs.
- OpenMP scales well up to a certain core count—**beyond which gains taper off**.
- CUDA offers **superior scalability and performance**, thanks to thousands of lightweight GPU threads.
- Real-world deployment (e.g., on RPIs or edge devices) benefits from having **both CPU and GPU options**.

---

## 💬 Feedback or Questions?

Feel free to reach out via issues or fork the project to explore further!

---
```
