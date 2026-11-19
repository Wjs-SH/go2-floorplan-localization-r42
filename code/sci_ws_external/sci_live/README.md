# 온보드 로봇 패키지 (2026-10-01)

두 개다. 서로 독립적으로 돈다.

| 폴더 | 하는 일 | 진입점 |
|---|---|---|
| **[map_gen/](map_gen/README.md)** | 안내도 사진 → 네비맵·의미지도·MCL 마스크·층 판별용 지도 | `python3 make_maps.py --building sw` |
| **[floor_id/](floor_id/README.md)** | 주행 데이터(누적 LiDAR + odom) → 지금 몇 층인지 | `python3 floor_id.py --building sw --data <주행폴더>` |

```
안내도 사진 ──map_gen──▶ out/<건물>/floor_id_maps/ ──복사──▶ floor_id/maps/ ──floor_id──▶ 층 판별
```

- 둘 다 **패키지 밖 경로를 읽지 않는다.** `/home` 을 통째로 가린 격리 환경(로봇 흉내)에서 실행해 확인했다.
- 둘 다 GPU·ROS·인터넷 불필요. Python 3.10 + numpy·scipy·opencv-contrib(4.x)·pillow·pyyaml.
- 처음 넣으면 각 폴더에서 `python3 check_env.py`.

## 지금 들어 있는 지도

- `floor_id/maps/` — 세 건물 24층. 소융대는 `map_gen` 결과와 같은 지도다.
  정보관·공대는 정리 단계(글자·아이콘 제거 등)가 생기기 전 파이프라인 지도이고, 층 판별 성적은 그 지도로 잰 것이다.
  `map_gen` 으로 만든 정보관·공대 지도로 바꾸려면 바꾼 뒤 성적을 다시 재야 한다.

## 공통 한계

- **미터 축척은 실측이 아니다**(복도 폭 3.3 m 를 빌려 추정). 미터 좌표는 실측 확인 전까지 믿지 말 것.
- 층 판별은 한 구간만으로 확정하지 말 것(1위 21%). 여러 구간을 모아 다수결로.
- 둘 다 ROS 노드가 아니다. 저장된 데이터·사진을 읽어 오프라인으로 돈다.
