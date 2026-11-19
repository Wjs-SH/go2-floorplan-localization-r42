#!/usr/bin/env python3
"""pt_console.py — 사용자 조준 콘솔(aim_console.py, 파일 수정 없음)을 명판 재촬영 사이트용으로 띄우는 실행기 (학생 스테이징 2026-11-13).

plate_site.py 가 [팬틸트 연결] 을 누를 때만 이 파일을 실행한다. 직접 실행하지 말 것: 콘솔은 /dev/ttyUSB0 를 열면서 나노를 리셋하고
(부팅 자세 -> 정면), 같은 포트를 두 프로세스가 열면 서로 리셋한다.
메모리 안에서만 바꾸는 것 세 가지(선배의 e6_aim_console_live2.py 와 같은 방법):
  1. 콘솔 HTTP 를 127.0.0.1 에만 연다 — 콘솔의 이동 명령에는 확인 절차가 없으므로 사이트(허용 주소 확인)를 거쳐서만 닿게 한다.
  2. RealSense 영상 토픽을 구독하지 않는다 — 영상 토픽을 읽는 프로세스가 하나 더 생기면 모든 프로세스의 정렬 깊이가 멎는다(11-01 현장, R34).
     사이트는 센서 허브의 장면을 쓰고, 조준 거리도 사이트가 준다.
  3. 팬틸트 카메라를 by-id 경로로 찾는다(이름 검색이 한 번 실패한 적이 있다, 10-28)."""
import os
import sys
from pathlib import Path

KC = Path(os.environ.get("PLATE_SITE_KC") or (Path.home() / "sci_ws/repairs/livefield2_20261018_151606/candidate/external/sci_event/code"))
CAM_BYID = "/dev/v4l/by-id/usb-Arducam_Technology_Co.__Ltd._Arducam_IMX477_HQ_Camera_UC517-video-index0"
sys.path.insert(0, str(KC))
import aim_console as AC  # noqa: E402  사용자 모듈, 수정 없음


def local_only(AC):
    cc = AC.cc
    base = cc.Server

    class LocalServer(base):
        def __init__(self, addr, handler, *a, **k):
            super().__init__(("127.0.0.1", addr[1]), handler, *a, **k)
    cc.Server = LocalServer


def no_realsense_images():
    try:
        from rclpy.node import Node
    except Exception as ex:                                  # noqa: BLE001  ROS 환경이 없으면 구독 자체가 안 생긴다
        print("plate_site: rclpy 없음 (%s) — RealSense 구독 없음" % type(ex).__name__, flush=True)
        return
    orig = Node.create_subscription

    def create_subscription(self, msg_type, topic, *a, **k):
        if getattr(msg_type, "__name__", "") == "Image" and str(topic).startswith("/camera/"):
            print("plate_site: %s 구독 안 함 (센서 허브만 영상을 읽는다)" % topic, flush=True)
            return None
        return orig(self, msg_type, topic, *a, **k)
    Node.create_subscription = create_subscription


def camera_byid(AC):
    orig = AC.find_camera

    def find_camera(*a, **k):
        return CAM_BYID if os.path.exists(CAM_BYID) else orig(*a, **k)
    AC.find_camera = find_camera


if __name__ == "__main__":
    local_only(AC)
    no_realsense_images()
    camera_byid(AC)
    raise SystemExit(AC.main())
