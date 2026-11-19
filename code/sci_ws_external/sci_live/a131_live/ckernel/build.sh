#!/bin/bash
# C++ 커널 만들기(노트북·로봇 공용). -ffp-contract=off: 곱셈-덧셈을 합치지 않아 numpy 와 끝자리까지 같게.
cd "$(dirname "$0")" && g++ -O3 -fPIC -shared -ffp-contract=off -o liba131_kernel.so a131_kernel.cpp && ls -la liba131_kernel.so
