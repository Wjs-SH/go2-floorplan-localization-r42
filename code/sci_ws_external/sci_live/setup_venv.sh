#!/usr/bin/env bash
# 이 폴더 안에만 파이썬 3.10 가상환경(venv/)을 만들고 필요한 라이브러리를 깐다.
# 시스템 파이썬·~/.local·~/.cache 는 건드리지 않는다(pip 캐시 끔). 인터넷 필요(한 번).
# 버전은 정보관·공대·소융대 지도를 만든 노트북과 같게 고정했다(결과를 픽셀까지 맞추려고).
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON310:-/usr/bin/python3.10}"
[ -x "$PY" ] || { echo "파이썬 3.10 이 없다: $PY (PYTHON310=경로 로 지정)"; exit 1; }
[ -d "$D/venv" ] || "$PY" -m venv "$D/venv"
PIP_NO_CACHE_DIR=1 "$D/venv/bin/pip" install --no-cache-dir --disable-pip-version-check -q \
  numpy==1.21.5 scipy==1.8.0 opencv-contrib-python-headless==4.5.4.60 pillow==12.3.0 pyyaml
"$D/venv/bin/python" -c "import numpy,scipy,cv2,PIL,yaml; print('numpy',numpy.__version__,'scipy',scipy.__version__,'cv2',cv2.__version__,'ximgproc',hasattr(cv2,'ximgproc'))"

# Jetson(Go2 Orin NX 등): 시스템 파이썬 3.8 에 CUDA torch 가 있으면 층 판별용 GPU 환경 venv_gpu 도 만든다.
# torch 는 시스템 것을 읽기만 하고(--system-site-packages), 나머지는 같은 버전을 venv_gpu 안에 깐다.
PY38="${PYTHON38:-/usr/bin/python3.8}"
if [ -x "$PY38" ] && PYTHONNOUSERSITE=1 "$PY38" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  [ -d "$D/venv_gpu" ] || "$PY38" -m venv --system-site-packages "$D/venv_gpu"
  PYTHONNOUSERSITE=1 PIP_NO_CACHE_DIR=1 "$D/venv_gpu/bin/pip" install --no-cache-dir --disable-pip-version-check -q \
    numpy==1.21.5 scipy==1.8.0 opencv-contrib-python-headless==4.5.4.60 pillow==10.4.0 pyyaml 2>&1 | grep -v "has requirement" || true
  PYTHONNOUSERSITE=1 "$D/venv_gpu/bin/python" -c "import torch,numpy,cv2; print('venv_gpu: torch',torch.__version__,'cuda',torch.cuda.is_available(),'numpy',numpy.__version__,'ximgproc',hasattr(cv2,'ximgproc'))"
else
  echo "CUDA torch 가 없다 — 층 판별은 venv(CPU)로 돈다(결과는 같고 느리다)."
fi
