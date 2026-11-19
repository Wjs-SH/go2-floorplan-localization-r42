# a1_3_1 라이브판 (2026-10-06)

10-06 요청서("a1_3_1 구조점 추출을 Go2 에서 실시간으로")에 대한 답입니다. 녹화본 없이 스캔 배열과 FAST-LIO 자세만으로 표본 하나(`a1_3_1_all_candidate_xy`, `a1_3_1_all_candidate_role`)와 원거리 점을 메모리 안에서 만듭니다. 서브프로세스와 디스크 쓰기가 없고 ROS 도 필요 없습니다(numpy, scipy 만).

계산은 전달본을 만든 코드(V0.4 오케스트레이터 + 빠른 판 `/data2/a131_src_fast`)를 그대로 불러 씁니다. 그래서 같은 입력이면 결과가 비트 단위로 같습니다. 바꾼 것은 단계 사이 파일 주고받기를 메모리로 돌린 것, rosbag 읽기 대신 배열을 받는 것, 최종 결과에 안 쓰이는 계산(그림, 384셀 선택, A1 구조 특징, 미리보기 선택)을 뺀 것입니다.

## 폴더

| 파일 | 내용 |
|---|---|
| `a131_live.py` | 라이브판 전부. `A131Live`, `far_points`, 자세 변환 함수 |
| `source/` | 불러 쓰는 기존 단계 코드 8개(손대지 않은 사본) |
| `demo_from_dump.py` | ROS 없이 시험 데이터로 돌려 기준 결과와 대조하고 속도를 잼 |
| `testdata/S6_chunk0_55.npz` | S6 녹화본 0~9초: 스캔 84개, 자세, 표본 55개의 기준 결과(33 MB) |
| `laptop_tools/` | 노트북에서 녹화본으로 대조할 때 쓴 스크립트(ROS 필요, 로봇에는 필요 없음) |

로봇에서는 이 폴더를 `~/jsh_onboard_robot_20261001/a131_live/` 에 두고 기존 `venv`(Python 3.10, numpy 1.21.5, scipy 1.8.0)로 돌리면 됩니다. 노트북에서 같은 버전으로 시험했습니다. 시스템 전역 설치는 필요 없습니다.

## 먼저 로봇에서 돌려 볼 것

```bash
cd ~/jsh_onboard_robot_20261001/a131_live
../venv/bin/python demo_from_dump.py testdata/S6_chunk0_55.npz --every 7 --workers 3 --fast-pca
```

`--every 7` 은 표본을 약 1초 간격으로, `--workers 3` 은 스캔 3개를 프로세스 3개로 동시에, `--fast-pca` 는 아래 빠른 면 방향 계산입니다. 표본마다 기준과 같은지, 다르면 몇 칸이 다른지와 걸린 시간이 나옵니다. Orin 에서 표본당 몇 초인지는 이걸로 재 주세요.

## 쓰는 법

```python
from a131_live import A131Live, make_interpolator, fastlio_to_base, fastlio_to_odom_link, FarPoses, far_points

L = A131Live(static_matrix=T_base_lidar, workers=3, exact_pca=False)   # T_base_lidar: base_link←hesai_lidar 4×4

# FAST-LIO /Odometry(camera_init→imu) 가 올 때마다 두 가지로 바꿔 버퍼에 쌓는다
p_base, q_base = fastlio_to_base(p_imu, q_imu, delta_deg=45.3)    # a1_3_1 용 odom→base_link
x, y, yaw = fastlio_to_odom_link(p_imu, q_imu)                    # 위치추정 입력 odom_link 규약

# 표본 하나 = 최근 스캔 3개(직전 2개 + target)
items = []
for sid, scan in last_three_scans:                                # scan: PointCloud2 를 푼 배열
    interp = make_interpolator(stamps_ns_win, p_base_win, q_base_win)   # 그 스캔 시작 −2초 ~ 끝(+다음 자세 1개) 구간
    items.append((sid, xyz_float64, intensity, ring, timestamp_sec, interp, None))
L.add_scans(items)                                                # 이미 계산한 스캔은 건너뜀
out = L.extract([sid0, sid1, sid_target], merge_interp)          # merge_interp: 세 스캔을 덮는 자세 구간
# out["xy"] (N,2) float32 몸체 기준 x 앞·y 왼쪽, out["role"] (N,) int8, out["stamp_ns"] = target 스캔 timestamp 최댓값

far = far_points(x_f32, y_f32, z_f32, ring, timestamp_sec, FarPoses(t_sec_buf, p_base_buf, q_base_buf))   # (M,2), 10점 미만이면 빈 배열
```

스캔 배열은 PointCloud2 필드 `x, y, z, intensity, ring, timestamp` 를 순서·개수 그대로(16 ring × column 격자, NaN 허용) 넘깁니다. 스캔 끝 뒤의 자세가 하나 더 있어야 끝부분을 제대로 보간합니다(없으면 0.75초까지 마지막 자세로 붙잡아 계산은 되지만 덜 정확). FAST-LIO 주기(10 Hz)만큼 0.1초쯤 기다린 뒤 부르면 됩니다.

## 자세 규약 (가장 중요)

전달본 a1_3_1 과 odom_link.csv 는 같은 FAST-LIO 출력(camera_init→imu) 하나에서 서로 다른 규약으로 만들어졌습니다. 로봇에서도 FAST-LIO 출력 하나에 두 변환을 각각 적용하면 됩니다.

| 쓰는 곳 | 변환 | 전달본과 대조 |
|---|---|---|
| a1_3_1 (deskew, 3스캔 합치기, 원거리 점) | `fastlio_to_base`: 레버암(imu→base 0.026, 0, −0.042) → y 거울(p.y, qx, qz 부호 반전) → 몸체 z 축 δ 회전 | S6 정제 녹화본 /tf 3807개와 위치 4e-16 m, 각도 0° |
| 위치추정 odom_link | `fastlio_to_odom_link`: x := −x, y 그대로, yaw := −yaw | S6 odom_link.csv 와 위치 0.0001 m 이하 |

δ 는 녹화마다 추정했는데 44.7~45.7°(6F 45.71, 8F 45.62, S1 45.28, S6 44.71)로 거의 같아서 로봇 장착 각도로 보고 45.3° 를 기본값으로 두었습니다. `estimate_delta_deg` 로 직진 구간(0.25 m/s 넘게, 0.5초 동안 회전 3° 미만)에서 다시 잴 수 있습니다(S6 로 다시 재면 44.67°).

위 규약은 노트북 FAST-LIO 설정 기준입니다. 로봇 FAST-LIO(`~/go2_ws/src/FAST_LIO_ROS2`) 의 IMU 축이나 frame 이 다르면 거울상과 δ 가 달라질 수 있습니다. 로봇에서 직진 몇 m 와 제자리 좌회전을 한 번 기록해서 두 가지를 확인해 주세요. `fastlio_to_base(…, delta_deg=0)` 결과로 `estimate_delta_deg` 가 45° 근처인지, 그리고 `fastlio_to_base` 결과의 yaw 가 좌회전에서 커지는지입니다.

## 기존 결과와 같은지

| 대조 | 결과 |
|---|---|
| 구조점, S6 원본 녹화본 0~9초 표본 55개 | 55개 비트 동일 |
| 구조점, S6 정제 녹화본(전달본 a131_S6_clean_v7) 180~189초 표본 60개 | 60개 비트 동일 |
| 원거리 점, 전달본 300개 | 300개 비트 동일, 하나에 4 ms |
| FAST-LIO 원본 → `fastlio_to_base` → 구조점 (로봇과 같은 경로), 60개 | 표본당 약 600칸 중 다른 칸 중앙 1, 최대 13 |
| 위와 같고 δ 45.3° 고정 + 빠른 면 방향 + 1초 간격, 9개 | 다른 칸 중앙 7, 최대 14(3.3% 이하) |

마지막 두 줄의 차이는 자세 보간 기준점 차이 때문입니다. 전달본은 라이다 헤더 시각에 미리 보간한 자세(/tf)로 만들었고, 라이브판은 FAST-LIO 주기 자세를 바로 보간합니다. 층 판별 성적이 유지되는지는 그쪽에서 재 주세요.

## 속도

노트북 CPU 1코어(프로세스당 BLAS 1스레드) 기준 단계별 시간입니다.

| 단계 | 스캔당 / 표본당 | 비고 |
|---|---|---|
| A1 (deskew) | 34 ms | 원본은 시각마다 전체 점을 훑었다(O(시각×점)). 같은 결과로 정렬해 묶음 |
| A1.1 (면 방향 PCA) | 470 ms, 빠른 판 230 ms | 가장 무거움. GPU 로 옮길 수 있는 부분 |
| A1.2 / A1.2.1 | 88 / 60 ms | |
| merge + A1.3 + 역할 | 표본당 140 ms | |

표본 간격이 0.3초보다 길면 표본마다 새 스캔 3개가 필요합니다. 1초 간격일 때 표본당 시간은 아래와 같습니다.

| | 1코어 | 3코어(`workers=3`) |
|---|---|---|
| 기존과 같은 계산 | 2.2초 | 1.1초 |
| 빠른 면 방향(`exact_pca=False`) | 1.4초 | 0.77초 |

Orin 이 노트북보다 2~3배 느리다면 3코어로 1.5~2.3초입니다. 빠른 면 방향은 이웃 모멘트 누적과 3×3 고유값 공식으로 계산하며, 스캔당 5만 8천 점 중 법선이 다른 점은 8개였습니다(이웃이 일직선이라 법선이 원래 정해지지 않는 점). 더 줄여야 하면 A1.1 을 GPU 로 옮기는 것이 다음 순서입니다(로봇 torch 사용 가능, 결과는 점 몇 개 차이).

## 전달본과 다른 점

- 전달본의 role=4 에는 오프라인 유령점 마스크(`apply_ghost_mask`, 녹화 전체로 만든 유령점 목록)가 덮어써져 있습니다. 라이브판에는 없습니다. 원래 role=4(6 m 넘고 불확실·지지 약함)만 있습니다.
- 전달 녹화본은 이중반사·따라오는 사람·움직이는 물체 점을 지운 정제본이었습니다. 라이브판은 들어온 스캔을 그대로 씁니다(정제 단계는 별도).
- numpy·scipy 판이 다르면 경계 칸 1~2개가 달라질 수 있습니다. 시험 데이터 기준 결과는 numpy 1.26.4/scipy 1.15.3 으로 만들었고, 로봇 venv 판(1.21.5/1.8.0)으로는 20개 중 16개 같음, 4개는 1칸 차이였습니다.

## 요청서 질문에 대한 답

1. 라이다 토픽: 녹화본 기준 `/lidar_points`, frame `hesai_lidar`, 필드 x·y·z(float32), intensity(float32), ring(uint16), timestamp(float64 절대초), point_step 26. 로봇 실물 토픽 이름과 드라이버 설정(`~/go2_ws` / `~/go2_lidar`)은 아직 로봇에서 확인하지 않았습니다.
2. FAST-LIO: 노트북에서는 `/Odometry`(camera_init→imu) 를 씁니다. 로봇의 실행 방법과 frame 이름은 로봇에서 확인이 필요합니다. 위 "자세 규약"의 확인을 같이 부탁합니다.
3. 원거리 점 코드: `/data2/rosbags/flower_usb/_a131_clean/extract_far_generic_deskew.py`(v2) 를 `far_points()` 로 옮겼습니다.
4. 코드 판: 재빌드 전달본(S6 v7t3, 4F v7tr2s, 7F v5m2 등)은 V0.4 stride015 오케스트레이터 + 빠른 판 트리(`/data2/a131_src_fast`, `_fast2`)입니다. 빠른 판은 원본 트리와 비트 동일함을 확인한 판입니다. 받으신 `vendor/a1_3_1` 은 빠른 판 이전의 원본 트리입니다. 계산 결과는 같고, 정제 녹화본(지운 점이 NaN)에서 스캔 시각 정의 한 곳만 다릅니다.
5. 4-2 단계별 무게와 GPU: 위 "속도" 표. 원래 표본당 8.8초 중 대부분은 단계마다 프로세스를 띄우고 npz 를 쓰고 읽는 데 들었습니다. S3 3스캔 합치기 자체는 50 ms 입니다.
