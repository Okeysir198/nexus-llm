# Build the PrismML llama.cpp fork (branch `prism`) with CUDA for GB10
# (aarch64, sm_121). Stock llama.cpp silently produces garbage on Bonsai
# PTQ1_0/PQ2_0 quants — this fork is required (model card, KNOWN_ISSUES.md).
#
# No prebuilt linux-cuda-aarch64 release exists (only x64 CUDA / arm64
# CPU+vulkan), so we build. Build is CPU-bound; run it with a CPU ceiling —
# a 20-core compile pushed this board past 90 C once already (see global
# CLAUDE.md thermal notes).
FROM nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      cmake build-essential libcurl4-openssl-dev \
 && rm -rf /var/lib/apt/lists/*

COPY llama.cpp-src /src
WORKDIR /src

# GG10/Spark = compute capability 12.1. Keep the build single-arch to halve
# compile time; -DLLAMA_CURL=ON enables HF fetch in llama-server (unused with
# local mounts but harmless).
RUN cmake -B build \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=121 \
      -DLLAMA_CURL=ON \
      -DCMAKE_BUILD_TYPE=Release \
 && cmake --build build --config Release --target llama-server -j"$(nproc)"

RUN cp build/bin/llama-server /usr/local/bin/llama-server

# Runtime entry: the engine compose passes the GGUF path + flags.
EXPOSE 8000
ENTRYPOINT ["llama-server"]
