"""FAST-LIO /Odometry(camera_init→imu, 거울상 규약) → 로봇 몸체(base) 평면 자세. 2026-10-15.

sci_event/code/calib_console.py on_odom(--odom-base, 2026-10-14)의 식을 함수로 뗀 것(같은 값). 실시간 노드의
a131_live.fastlio_to_base 와 같은 순서(레버암 → y 거울상 → 몸체 z 축 δ 회전)를 평면에서 한다.
선배 e5_node 는 오돔을 'x 전진·y 왼쪽, RealSense 가 로봇 정면' 으로 쓰는데 /Odometry 원본은 거울상이고 전진축이 δ 만큼 돌아 있다.
"""
import math

LEVER = (0.026, 0.0)          # imu → base_link 레버암 x·y (a131_live.LEVER_IMU_BASE)


def fastlio_to_base_2d(x, y, yaw, delta_deg=45.3, lever=LEVER):
    lx, ly = lever
    dl = math.radians(delta_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    bx, by = x + c * lx - s * ly, -(y + s * lx + c * ly)
    ys = -yaw
    yaw2 = ys + dl
    bx += -(math.cos(ys) * lx - math.sin(ys) * ly) + (math.cos(yaw2) * lx - math.sin(yaw2) * ly)
    by += -(math.sin(ys) * lx + math.cos(ys) * ly) + (math.sin(yaw2) * lx + math.cos(yaw2) * ly)
    return bx, by, yaw2
