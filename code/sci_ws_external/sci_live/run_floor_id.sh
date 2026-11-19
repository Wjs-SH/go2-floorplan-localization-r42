#!/usr/bin/env bash
# 이 폴더의 가상환경으로 floor_id/floor_id.py 를 돌린다. 인자는 그대로 넘긴다(예: --building sw --data ...).
# venv_gpu(Jetson: 시스템 torch CUDA 사용)가 있으면 그것을, 없으면 venv(CPU)를 쓴다.
# 벽 정합이 GPU 로 돌면 Go2(Orin NX)에서 시점 하나에 수 분, CPU 만이면 약 27분 걸린다(결과는 같다).
# 로봇의 다른 작업을 방해하지 않도록 낮은 우선순위(nice 10)로 돈다. NICE=0 으로 바꿀 수 있다.
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${PY:-}" ]; then
  if [ -x "$D/venv_gpu/bin/python" ]; then PY="$D/venv_gpu/bin/python"; else PY="$D/venv/bin/python"; fi
fi
[ -x "$PY" ] || { echo "가상환경이 없다 — 먼저 ./setup_venv.sh"; exit 1; }
export PYTHONNOUSERSITE=1     # ~/.local 의 패키지를 섞지 않는다
exec nice -n "${NICE:-10}" "$PY" "$D/floor_id/floor_id.py" "$@"
