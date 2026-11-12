#!/bin/bash
# Run inside the nsight pod: builds NVIDIA Nsight Perf SDK for AI into /work/sdk/build/bin and
# copies the image's CUDA 13.5 compatibility libraries to /work/compat for the vLLM pod.
# The proprietary service (nvidia-nsight-perf) is downloaded by the SDK's CMake from the public
# GitHub release of NVIDIA/nsight-perf-daemon: no NGC login needed.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git cmake build-essential ca-certificates curl zip unzip python3 >/dev/null
mkdir -p /work && cd /work
[ -d sdk ] || git clone --depth 1 https://github.com/NVIDIA/nsight-perf-sdk-for-ai.git sdk
cd sdk && git rev-parse HEAD && git submodule update --init --recursive --depth 1
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > /work/cmake-configure.log 2>&1 || { tail -40 /work/cmake-configure.log; exit 1; }
cmake --build build -j "$(nproc)" > /work/cmake-build.log 2>&1 || { tail -60 /work/cmake-build.log; exit 1; }
ls -l build/bin
mkdir -p /work/compat
cp -av /usr/local/cuda*/compat/. /work/compat/ 2>/dev/null || echo "no compat directory in this image"
