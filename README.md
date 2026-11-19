# Go2 평면도 기반 층·자세 위치추정 (R42, E1–E8)

Unitree Go2 사족 로봇이 **몇 층의 어디에, 어느 방향으로** 있는지를 스스로 찾는 시스템의 운영 코드다. 출발 위치와 층은 모르는 상태에서 시작한다. 쓰는 정보는 세 가지다.
- 건물 안내도로 만든 평면도(벽·방·방 번호)
- LiDAR 관성 주행거리계(FAST-LIO)
- 복도 명판의 방 번호(카메라 탐지 + 팬틸트 사진 판독)

이 저장소는 로봇에서 돌고 있는 판을 그대로 옮긴 것이다. 함께 실린 것은 다른 사람이 구조를 분석할 수 있도록 고른 설명 문서와 실험 기록 1건이다.

> **English summary.** Operational code of a floor-and-pose localization system for a Unitree Go2 quadruped in a 9-floor building. It starts with an unknown floor and an unknown start pose. It uses three sources:
> - a floor plan generated from the building's directory signs,
> - FAST-LIO LiDAR-inertial odometry,
> - room-number plates, detected zero-shot with SigLIP and read by OCR from a pan-tilt photo.
>
> The pipeline has eight stages, E1–E8: maps, all-floor coarse search, tracking, ambiguity, plate detection, plate reading, plate-anchored registration and floor decision, and pairing/display. The full Korean walk-through is in `docs/`. Floor plans and model weights are **not** included.

## 구성

| 경로 | 내용 |
|---|---|
| `docs/R42_E1_E8_SYSTEM_KO.md` | (다음 커밋에서 추가) **시스템 전체 설명.** 프로세스와 통신, E1~E8 단계별 알고리즘, 확정 조건, 실험 한 건을 처음부터 확정까지 따라간 기록 |
| `code/robot_live2/` | 로봇 운영 트리 `~/sci_ws/repairs/livefield2_20261018_151606/` 중 코드·설정·시험·문서. 진입점은 `candidate/live2/console.py` |
| `code/robot_live2/candidate/live2/` | E2~E8 본체(콘솔·감독, E2 탐색, E3 추적, E5 탐지·정지, E6 판독, E7 정합·층 판정, E8 쌍 판정) |
| `code/robot_live2/candidate/live2/web_v2/` | 운영 화면(태블릿 브라우저) |
| `code/robot_live2/candidate/src/` | 관측·판독·팬틸트·의미 위치추정 ROS 2 패키지 |
| `code/robot_live2/config/e7_semantic_policy.json` | 의미 확정 정책. 없으면 의미 확정이 SHADOW 로만 돈다 |
| `code/sci_ws_external/` | 로봇 `~/sci_ws/external/` 중 코드. 운영 코드가 이 경로를 직접 읽는다(`floor_id`, `sci_live`, `sci_event`). FAST-LIO 사본 2개 포함(GPL-2.0, 각 폴더의 LICENSE) |
| `code/robot_live2/candidate/external/` | 위 `external` 의 예전 사본. 운영 중에는 쓰이지 않으며, 비교용으로 남겼다 |
| `code/home_extra/` | 운영 코드가 홈 폴더에서 읽는 파일. 팬틸트 보정(`pantilt_calib/configs/pantilt_calib_multi_20260928.json`), E2 스위치 파일(`kci_loadtest_20261113/e2_acc.on`, `e2_clean.on`) |
| `examples/X20261116_182124/` | 현장 실험 기록 1건(정답 5층, 명판 502 → 504). 문서의 예시가 이 기록이다 |

운영 중 반영된 판은 아래 sha256 앞 12자리로 확인할 수 있다(`code/robot_live2/candidate/live2/`).

| 파일 | sha256 | 내용 |
|---|---|---|
| `e6_semantic.py` | 183f947eb11e | 명판 위치 검사 — 복도 반폭 규칙 |
| `e8_pair.py` | db6cf115c213 | 쌍 판정 + 경로 기준 후보 |
| `semantic_anchor.py` | 5c7f01cc8327 | 명판 기준 후보 |
| `e6e7_link.py` | 54e548e1f400 | E6–E7 연결 |
| `e7_proc.py` | 8b16887092b5 | E7 별도 프로세스 |
| `fused_floor.py` | 240ae84565dc | 층 판정(FUSED_FLOOR v3) |
| `e3_tracker.py` | ac5957eadd4c | E3 추적 |

## 저장소에 없는 것 (실행에 필요)

이 저장소만으로는 로봇 위에서 돌릴 수 없다. 코드가 읽는 자리와 필요한 것은 다음과 같다.

| 필요한 것 | 로봇에서의 위치 | 비고 |
|---|---|---|
| 층별 평면도 | `~/sci_ws/maps/kci_handoff/{n}F/` | `{n}F_semantic_map.json` 의미지도, `{n}F_base_map.png` 구조지도, E3 용 경량 지도. 건물 내부 정보와 방 이름(사람 이름 포함)이 있어 싣지 않았다 |
| SigLIP | `~/sci_ws/models/siglip-base-patch16-224` | Hugging Face `google/siglip-base-patch16-224` 그대로(추가 학습 없음, 문구 3개로 사용) |
| VLM | `~/sci_ws/models/Qwen3-VL-2B-Instruct-GGUF` | Qwen3-VL-2B-Instruct Q8_0 GGUF, llama.cpp 서버로 실행. llama.cpp 커밋 `318a471a6295e53bfce7ca96ae0cf26071cbdd6e` |
| OCR | 파이썬 환경 | RapidOCR(PP-OCRv4). `~/sci_ws/models/easyocr` 는 비교 시험용 |
| ROS 2·FAST-LIO 빌드 | `~/sci_ws/external/sci_live/fastlio_ws` 등 | 소스는 `code/sci_ws_external/` 에 있다. 빌드 산출물은 없다 |
| 가상환경 | `~/sci_ws/.venv`, `sci_live/venv_gpu` | `code/sci_ws_external/sci_live/setup_venv.sh`, `INSTALL_*.txt` 참고 |

## 실험 기록 예시 (`examples/X20261116_182124`)

- **들어 있는 것**
  - 실험 설정과 시간 기록: `experiment.json`, `timeline.jsonl`
  - 프로세스별 출력·감독 기록: `*.out`, `*.sup.jsonl`
  - 단계별 상태: `state/`
  - LiDAR 표본: `archive/live2_store/` 의 `odom_link.csv` 와 `samples/*/data_short.npz`
  - 관측 노드 기록: `archive/observe_run/`
- **사진**: 앞 카메라 장면, 명판 사진, 화면 캡처 등 540장은 모두 뺐다(건물 내부). `archive/MANIFEST.json`·`SHA256SUMS` 에는 원래 파일 목록이 그대로 있다.
- **이 기록이 만들어진 판**: 현장 실험 당시(11-16) 판이라 현장 기록에는 공식 확정이 없다. 지금 운영판으로 같은 기록을 다시 판정한 결과는 문서 9장에 있다(259.2 s 쌍 판정 → E7 → 5층 확정).

## 공개본에서 바꾼 것

- **사람 이름**
  - 판독 결과·시험 코드·지도 글자 목록의 사람 이름은 가명(`가나다` 등)이나 `○` 로 바꿨다.
  - 기록의 방 이름 칸은 방 번호만 남겼다(`"504 [room name removed]"`).
  - 이 때문에 `tests/test_r36.py` 등 일부 시험은 실제 지도와 이름이 맞지 않는다. 지도도 없으므로 그대로 돌리면 실패한다.
- **사설 접속 주소**: Tailscale 주소·호스트 이름은 `<ROBOT_TAILSCALE_IP>` 등으로 바꿨다.
- **빠진 파일**: 인증 정보(`runtime/ctl/auth/` 등), 운영 상태(`runtime/`), 실험 기록 전체(`experiments/`), 로그, 예전 판 백업(`*.before_*`)은 처음부터 복사하지 않았다.

## 읽는 순서

1. `docs/R42_E1_E8_SYSTEM_KO.md` 1~2장: 무엇을 하는가, 어떤 프로세스가 무엇으로 통신하는가.
2. 같은 문서 3~8장: 단계별 알고리즘. 각 절에 코드 위치(`파일:줄`)가 있다.
3. 같은 문서 9장: `examples/X20261116_182124` 를 시간순으로 따라가며 각 파일에서 무엇을 보면 되는지.
