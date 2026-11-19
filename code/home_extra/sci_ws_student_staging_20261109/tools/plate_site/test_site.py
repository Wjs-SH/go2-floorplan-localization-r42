#!/usr/bin/env python3
"""test_site.py — 명판 재촬영 사이트 시험 (장치·실험 없이: 가짜 허브·가짜 SigLIP 상태·가짜 조준 콘솔). 로봇의 /usr/bin/python3 로 실행.
   python3 test_site.py"""
import json, math, os, sys, tempfile, time
from pathlib import Path
import numpy as np

TMP = Path(tempfile.mkdtemp(prefix="plate_site_test_"))
os.environ.update(PLATE_SITE_O=str(TMP / "O"), PLATE_SITE_S=str(TMP / "S"), PLATE_SITE_HUB=str(TMP / "hub"), PLATE_SITE_PT_PORT="18089", PLATE_SITE_FAKE_CONSOLE="1",
                  PLATE_SITE_STORE=str(TMP / "store"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import aimgeo, fake_console, plate_site as PS
G = aimgeo.G
K = [606.7977294921875, 0.0, 310.4879150390625, 0.0, 606.7450561523438, 240.51687622070312, 0.0, 0.0, 1.0]
for d in ("O/runtime/ctl", "O/runtime/e5_shm", "O/runtime/fastlio/Log", "hub", "S"):
    (TMP / d).mkdir(parents=True, exist_ok=True)
PS.GEO = geo = aimgeo.Geo()
N = dict(ok=0, seq=100)


def check(name, cond, detail=""):
    print(("  통과  " if cond else "  실패  ") + name + ("  " + str(detail) if detail else ""))
    if not cond:
        raise SystemExit("실패: " + name)
    N["ok"] += 1


def world(u, v, h_deg, moving=False, B=2.0, consent=False, regions=(), depth_mm=2000, new_scene=True, exp_id="X_TEST", requests=None):
    """가짜 센서 한 벌을 쓴다: 실험 상태, 허브(위치 3초치 + 장면 1장), FAST-LIO 상태 기록, SigLIP 장면 (requests = E5 정지 요청 목록)"""
    now = time.time(); h = math.radians(h_deg)
    exp = dict(state="RUNNING", exp_id=exp_id)
    if consent:
        exp["pantilt_consent"] = dict(file="x")
    (TMP / "O/runtime/ctl/experiment.json").write_text(json.dumps(exp))
    rows = []
    for i in range(31):
        t = now - 3.0 + i * 0.1; uu = u - (3.0 - i * 0.1) * 0.5 if moving else u
        x, y, yaw = G.planar_to_raw(uu, v, h); rows.append([t, x, y, yaw])
    N["seq"] += 1
    (TMP / "hub/slot_00.color").write_bytes(np.full((480, 640, 3), 120, np.uint8).tobytes()); (TMP / "hub/slot_00.depth").write_bytes(np.full((480, 640), depth_mm, np.uint16).tobytes())
    (TMP / "hub/index.json").write_text(json.dumps(dict(odom=rows, odom_last_rx_wall=now, caminfo=dict(k=K, w=640, h=480),
                                                        frames=[dict(seq=N["seq"], stamp=now - 0.05, slot=0, w=640, h=480, depth_stamp=now - 0.05)])))
    (TMP / "O/runtime/fastlio/Log/mat_pre.txt").write_text("1.0 -2.0 %.2f -75.0 0 0 0\n" % B)
    if new_scene or requests is not None:
        (TMP / "O/runtime/e5_shm/status.json").write_text(json.dumps(dict(siglip="UP", scene=dict(stamp=now - 0.1, thr=0.10, regions=list(regions)),
                                                                          live2_stop_request=dict(last=list(requests or [])))))
    PS.poll_once()


def plate_region(u, v, h_deg, tu, tv, up, pitch=0.0, sim=0.14, state="VERIFIED"):
    fwd, left = G.planar_to_body((u, v, math.radians(h_deg)), tu, tv)
    px = geo.pixel(fwd, left, up, K, pitch)
    return dict(box=[px[0] - 12, px[1] - 8, px[0] + 12, px[1] + 8], sim=sim, z=px[2], state=state, display=state)


def wait_pt(t=10.0):
    t0 = time.time()
    while PS.PTC.busy or PS.PTC.starting:
        if time.time() - t0 > t:
            raise SystemExit("팬틸트 작업이 끝나지 않음")
        time.sleep(0.05)


print("[1] 조준 기하")
for f, l in ((3.0, 1.2), (2.0, -1.0), (4.5, 0.3)):
    a = geo.servo(f, l, 1.72); b = G.calibrated_aim(*G.body_to_pixel(f, l, 1.72, K))
    check("3차원 식 = 운영 경로 (앞 %.1f 옆 %+.1f)" % (f, l), abs(a["pan_cmd"] - b["pan_cmd"]) < 0.02 and abs(a["tilt_cmd"] - b["tilt_cmd"]) < 0.02, (round(a["pan_cmd"], 2), b["pan_cmd"]))
worst = 0.0
for pan in range(25, 181, 5):
    for tilt in range(105, 181, 5):
        u_, v_, z_, err = geo.virtual_pixel(float(pan), float(tilt)); worst = max(worst, err)
check("가상 화소로 보낸 명령이 그대로 되돌아옴 (전 범위)", worst < 0.1, "최대 오차 %.3f도" % worst)
check("가장 가까운 지점: 왼쪽 1.2 m -> 앞 약 0.0 m", abs(geo.closest_fwd(1.2, 1.72) - 0.0) <= 0.051, geo.closest_fwd(1.2, 1.72))
check("가장 가까운 지점: 오른쪽 1.2 m -> 앞 약 0.35 m", abs(geo.closest_fwd(-1.2, 1.72) - 0.35) <= 0.051, geo.closest_fwd(-1.2, 1.72))
check("예비 위치가 범위를 넘으면 거부", not geo.check(22.0, 150.0)[0] and not geo.check(100.0, 103.0)[0] and geo.check(25.0, 105.0)[0])

print("[2] 케이스 1: 기억 -> 안내 -> 조준 -> 미세 조정")
fake_console.start(18089, TMP / "console")
TU, TV, UP = 6.0, 1.2, 1.72
for _ in range(3):
    world(2.0, 0.0, 0.0, regions=[plate_region(2.0, 0.0, 0.0, TU, TV, UP, state="SEARCH")])
check("E5 가 확인하지 않은 영역(SEARCH)은 자동으로 기억하지 않음", PS.ST["target"] is None)
FAR_ERR = -0.30                                      # 멀리서(4 m) 본 깊이가 0.3 m 짧게 나왔다고 치자
world(2.0, 0.0, 0.0, regions=[plate_region(2.0, 0.0, 0.0, TU + FAR_ERR, TV, UP)])
check("한 장면만으로는 기억하지 않음", PS.ST["target"] is None)
world(2.0, 0.0, 0.0, regions=[plate_region(2.0, 0.0, 0.0, TU + FAR_ERR, TV, UP)])
tg = PS.ST["target"]
check("두 장면 연속이면 자동 기억", tg is not None and tg["src"] == "SigLIP 자동")
check("기억한 위치 = 그때 본 위치(오차 포함)", math.hypot(tg["uv"][0] - (TU + FAR_ERR), tg["uv"][1] - TV) < 0.05 and abs(tg["up"] - UP) < 0.05, ([round(x, 3) for x in tg["uv"]], round(tg["up"], 3)))
world(1.5, 0.0, 0.0, regions=[plate_region(1.5, 0.0, 0.0, TU + 0.4, TV, UP)])
check("더 멀리서 본 값으로는 바꾸지 않음", abs(PS.ST["target"]["uv"][0] - (TU + FAR_ERR)) < 0.05 and PS.ST["target"]["n_refine"] == 0)
world(3.0, 0.0, 0.0, regions=[plate_region(3.0, 0.0, 0.0, TU + 0.2, TV, UP, state="REJECT")])
check("기각된 영역으로는 갱신하지 않음", PS.ST["target"]["n_refine"] == 0)
world(3.0, 0.0, 0.0, regions=[plate_region(3.0, 0.0, 0.0, TU, TV, UP)])
tg = PS.ST["target"]
check("더 가까이서 다시 보이면 기억 위치 갱신", math.hypot(tg["uv"][0] - TU, tg["uv"][1] - TV) < 0.05 and tg["n_refine"] == 1 and tg["best_rng"] < 3.4, ([round(x, 3) for x in tg["uv"]], round(tg["best_rng"], 2)))
world(2.0, 0.0, 0.0, new_scene=False)
g = PS.ST["guide"]; fmin = geo.closest_fwd(1.2, UP)
check("걷는 중 안내: 정지 지점(표지 1.0 m 앞)까지 남은 거리", g["phase"] == "walk" and abs(g["remaining"] - (4.0 - 1.0)) < 0.06 and g["stop_at"] == 1.0, g["title"])
world(TU - 0.3, 0.0, 0.0, new_scene=False)
g = PS.ST["guide"]
check("정지 지점을 지났지만 팬틸트가 닿는 곳: 지났다고 알리되 조준은 허용", g["phase"] == "over" and g["need"]["ok"] and abs(g["remaining"] + 0.7) < 0.06, g["title"])
STOP_U = TU - 1.02                                   # 표지를 1.02 m 앞에 둔 자리
world(STOP_U, 0.0, 0.0, new_scene=False)
g = PS.ST["guide"]
check("정지 지점 도착", g["phase"] == "stop" and g["need"]["ok"] and g["need"]["side_deg"] < 50, (g["title"], "팬틸트 옆 %s도" % g["need"]["side_deg"]))
check("팬틸트 미연결이면 조준 불가", not g["can_aim"] and not PS.op_aim()[0])
PS.PTC.connect(); wait_pt(); world(STOP_U, 0.0, 0.0, new_scene=False)
check("연결 뒤 조준 가능", PS.PTC.connected and PS.ST["guide"]["can_aim"])
ok, why = PS.op_aim(); wait_pt()
want = geo.servo(1.02, 1.2, UP); ph = PS.ST["photo"]
check("조준 + 촬영: 사진 저장", ok and ph is not None and (TMP / "S/_runs/plate_site/X_TEST/photos" / ph["name"]).is_file(), why or ph["name"])
check("보낸 명령 = 기억 위치 방향", abs(PS.PTC.cmd[0] - want["pan_cmd"]) < 0.3 and abs(PS.PTC.cmd[1] - want["tilt_cmd"]) < 0.3, (PS.PTC.cmd, round(want["pan_cmd"], 1), round(want["tilt_cmd"], 1)))
# ---- 사진 -> 운영판의 판독·위치추정 (e6_precision_results.jsonl 한 줄 + e6_captures 사진)
import hashlib
check("명판 촬영 방식 스위치가 꺼져 있으면 위치추정에 넘기지 않음", PS.op_use()[0] is False)
(TMP / "O/runtime/plate_case1.on").write_text("on")
world(STOP_U, 0.0, 0.0, new_scene=False); PS.poll_once()
_u = PS.op_use()
check("동의 없이 시작한 실험이면 넘기지 않음", _u[0] is False and "체크하지 않고" in _u[1], _u)
world(STOP_U, 0.0, 0.0, consent=True, new_scene=False); PS.poll_once()
_u = PS.op_use()
_pr = TMP / "O/runtime/state/e6_precision_results.jsonl"
_rec = json.loads(_pr.read_text().splitlines()[-1]) if _pr.exists() else {}
check("넘기면 운영판 형식의 한 줄이 생김", _u[0] and _rec.get("kind") == "LIVE2_E6_PRECISION" and _rec.get("status") == "CAPTURE_STORED" and _rec.get("experiment_id") == "X_TEST"
      and _pr.read_text().endswith("\n") and len(_pr.read_text().splitlines()) == 1, _u)
check("사진이 e6_captures 에 있고 지문이 맞음", Path(_rec["capture_file"]).is_file() and Path(_rec["capture_file"]).parent.name == "e6_captures"
      and hashlib.sha256(Path(_rec["capture_file"]).read_bytes()).hexdigest() == _rec["image_sha256"])
_xl, _yl = G.odom_link_point(*PS.ST["target"]["uv"])
check("표지 위치가 선배 E6 와 같은 좌표(odom_link)", abs(_rec["target_odom_link_xy"][0] - _xl) < 2e-3 and abs(_rec["target_odom_link_xy"][1] - _yl) < 2e-3, _rec["target_odom_link_xy"])
_x, _y, _yaw = G.planar_to_raw(STOP_U, 0.0, 0.0); _st = _rec["odom_pose_at_stop"]
check("촬영한 자리가 FAST-LIO 원래 값으로 들어감", abs(_st["x"] - _x) < 0.02 and abs(_st["y"] - _y) < 0.02 and abs(_st["yaw"] - _yaw) < 0.01 and abs(_st["t"] - time.time()) < 30
      and abs(_rec["stopped_wall"] - time.time()) < 60 and _rec["capture_wall"] >= _rec["stopped_wall"], _st)
check("운영판이 읽는 칸이 다 있고 HOME 확인은 주장하지 않음", all(k in _rec for k in ("observation_id", "capture_file", "image_sha256", "resolution", "image_quality", "target_odom_link_xy",
      "odom_pose_at_stop", "stopped_wall", "capture_wall", "wall", "ocr_lines", "vlm")) and _rec["image_quality"]["ok"] is True and "home_confirmed" not in _rec and _rec["ocr_lines"] == [])
check("저장소가 없으면 표지 위치를 그대로 두고 그 사실을 적음", _rec["store_frame"]["applied"] is False and _rec["store_frame"]["why"] == "NO_STORE"
      and _rec["target_odom_link_xy_fastlio"] == _rec["target_odom_link_xy"], _rec.get("store_frame"))
check("같은 사진을 두 번 넘기지 않음", PS.op_use()[0] is False and len(_pr.read_text().splitlines()) == 1)
(TMP / "O/runtime/state/e8_pair_state.json").write_text(json.dumps(dict(state="OBS1_WAITING_RESUME", why="x", written_wall=time.time(),
    observations=[dict(observation_id=_rec["observation_id"], read=True, ocr_lines=[["416", 0.97]])], evaluation=dict(verdict="PROVISIONAL", floors_explaining=[4]))))
_pv = PS.state_view()["use"]["pair"]
check("운영판의 판독·판정을 읽어 보여 줌", _pv["rows"][0]["id"] == _rec["observation_id"] and _pv["rows"][0]["read"] and _pv["rows"][0]["ocr"] == [["416", 0.97]]
      and _pv["verdict"] == "PROVISIONAL" and _pv["floors"] == [4], _pv)
(TMP / "O/runtime/plate_case1.on").unlink(); world(STOP_U, 0.0, 0.0, new_scene=False); PS.poll_once()
c0 = PS.PTC.cmd; ok, why = PS.op_nudge(5.0, 0.0); wait_pt()
check("오른쪽 5도 미세 조정: 팬 명령이 5 x 1.131 줄어듦", ok and abs((c0[0] - PS.PTC.cmd[0]) - 5.0 * geo.A.pan_gain) < 0.3 and abs(c0[1] - PS.PTC.cmd[1]) < 0.3, PS.PTC.cmd)
ok, why = PS.pt_op(dict(op="nudge", dx=-40.0, dy=0.0)); wait_pt()
check("범위를 넘는 이동은 보내지 않음", not ok and "범위" in (why or ""), why)
world(STOP_U, 0.0, 0.0, moving=True, new_scene=False)
ok, why = PS.op_aim()
check("로봇이 움직이면 조준 거부", not ok and "멈춘" in (why or ""), why)
world(TU + 0.6, 0.0, 0.0, new_scene=False)
check("지나치면 뒤로 안내", PS.ST["guide"]["phase"] == "passed", PS.ST["guide"]["title"])
n_ev = sum(1 for _ in open(TMP / "S/_runs/plate_site/X_TEST/events.jsonl"))
check("기록 파일에 남음 (기억·갱신·연결·촬영 2회)", n_ev >= 5, "%d줄" % n_ev)
fr_files = sorted(p.name for p in (TMP / "S/_runs/plate_site/X_TEST/frames").glob("*"))
check("기억·조준 순간의 고정 카메라 장면(컬러+깊이) 저장", len(fr_files) >= 6 and any("remember" in f for f in fr_files) and any(f.endswith("_depth.png") for f in fr_files), fr_files[:4])
PS.op_done(); wait_pt()
check("표지 끝 -> 정면 복귀, 기억 지움", PS.ST["target"] is None and tuple(PS.PTC.cmd) == tuple(geo.home))

print("[3] 케이스 2: 옆까지 -> 회전 -> 머리 들기 -> 화면에서 다시 찾아 조준")
PS.ST.update(case=2, auto=False, done=[])
TU2, TV2 = 6.0, 2.2                                  # 넓은 복도: 표지가 2.2 m 옆
world(0.0, 0.0, 0.0, regions=[plate_region(0.0, 0.0, 0.0, TU2, TV2, UP)])
r0 = PS.ST["scene"]["regions"][0]["box"]
ok, why = PS.op_pick((r0[0] + r0[2]) / 2, (r0[1] + r0[3]) / 2, PS.ST["rs_seq"])
check("상자를 눌러 기억", ok and PS.ST["target"] is not None and math.hypot(PS.ST["target"]["uv"][0] - TU2, PS.ST["target"]["uv"][1] - TV2) < 0.05, why)
PS.poll_once()
check("A: 표지 옆까지 전진", PS.ST["guide"]["phase"] == "A", PS.ST["guide"]["title"])
world(5.8, 0.0, 0.0, new_scene=False)
check("B: 표지 쪽으로 회전 안내(왼쪽)", PS.ST["guide"]["phase"] == "B" and "왼쪽" in PS.ST["guide"]["title"], PS.ST["guide"]["title"])
face = math.degrees(math.atan2(TV2, TU2 - 5.8))
for _ in range(8):
    world(5.8, 0.0, face, new_scene=False)
check("C: 정면 -> 머리 들기 안내", PS.ST["guide"]["phase"] == "C" and PS.ST["guide"]["seen_when_raised"], PS.ST["guide"]["title"])
reg = plate_region(5.8, 0.0, face, TU2, TV2, UP, pitch=18.0)
world(5.8, 0.0, face, B=-16.0, regions=[reg])
check("D: 머리 든 각도 인식 (약 18도)", PS.ST["guide"]["phase"] == "D" and 16.0 <= PS.ST["pitch_up"] <= 19.0, PS.ST["pitch_up"])
check("2.2 m 에서 머리를 들면 표지가 고정 카메라 화면 안", 0 <= (reg["box"][1] + reg["box"][3]) / 2 <= 480, [round(v) for v in reg["box"]])
ok, why = PS.op_aim(); wait_pt(); ph = PS.ST["photo"]
exp_cmd = geo.pixel_servo((reg["box"][0] + reg["box"][2]) / 2, (reg["box"][1] + reg["box"][3]) / 2, reg["z"])
check("지금 화면의 SigLIP 영역으로 조준", ok and ph["basis"].startswith("SigLIP 영역") and abs(PS.PTC.cmd[0] - exp_cmd[0]) < 0.5 and abs(PS.PTC.cmd[1] - exp_cmd[1]) < 0.5, (ph["basis"], PS.PTC.cmd))
world(5.8, 0.0, face, B=-16.0, regions=[], depth_mm=2200)
ok, why = PS.op_aim(); wait_pt(); ph = PS.ST["photo"]; pred = geo.servo(*G.planar_to_body(PS.ST["pose"], TU2, TV2), UP, PS.ST["pitch_up"])
check("영역이 없으면 기억 위치 + 몸 기울기로 조준", ok and ph["basis"].startswith("기억한 위치(몸 기울기") and abs(PS.PTC.cmd[1] - pred["tilt_cmd"]) < 0.5 and abs(PS.PTC.cmd[1] - exp_cmd[1]) < 3.0,
      (ph["basis"], [round(v, 1) for v in PS.PTC.cmd], "영역 조준 때 틸트 %.1f" % exp_cmd[1]))
ok, why = PS.op_pick(400.0, 200.0, PS.ST["rs_seq"])
check("상자 밖을 누르면 그 점을 조준점으로", ok and PS.ST["sel"] is not None and abs(PS.ST["sel"]["z"] - 2.2) < 0.01, PS.ST["sel"])
ok, why = PS.op_aim(); wait_pt()
check("고른 점으로 조준", ok and PS.ST["photo"]["basis"] == "화면 선택(점)", PS.ST["photo"]["basis"])
PS.op_done(); wait_pt()
world(5.8, 0.0, 0.0, new_scene=False)
PS.ST.update(case=2); world(0.0, 0.0, 0.0, regions=[plate_region(0.0, 0.0, 0.0, 6.0, 1.2, UP)])
r0 = PS.ST["scene"]["regions"][0]["box"]; PS.op_pick((r0[0] + r0[2]) / 2, (r0[1] + r0[3]) / 2, PS.ST["rs_seq"])
face2 = math.degrees(math.atan2(1.2, 0.2))
for _ in range(6):
    world(5.8, 0.0, face2, new_scene=False)
check("1.2 m 에서는 머리를 들어도 화면 밖이라고 알려 줌", PS.ST["guide"]["phase"] == "C" and PS.ST["guide"]["seen_when_raised"] is False and "벗어납니다" in PS.ST["guide"]["sub"], PS.ST["guide"]["sub"][-48:])

PS.ST["ahead"] = 0.0
world(5.8, 0.0, 0.0, new_scene=False); PS.ST.update(case=1); PS.poll_once()
check("정지 지점을 0 m 로 두면 팬틸트가 닿는 한계 바로 앞까지 안내(예전 동작)", abs(PS.ST["guide"]["stop_at"] - (geo.closest_fwd(PS.ST["guide"]["left"], UP) + 0.10)) < 0.06, PS.ST["guide"]["stop_at"])
PS.ST["ahead"] = 1.0
print("[4] 보호 장치")
world(5.8, 0.0, face, consent=True, new_scene=False)
check("팬틸트 동의로 시작한 실험이면 연결 거부 사유", any("팬틸트 카메라 사용" in w for w in PS.PTC.guard()))
(TMP / "O/runtime/plate_case1.on").write_text("on")                     # 명판 촬영 방식 스위치
world(5.8, 0.0, face, consent=False, new_scene=False); PS.poll_once()
check("명판 촬영 방식: 동의 없이 시작한 실험이면 연결 거부", any("체크하지 않고" in w for w in PS.PTC.guard()), PS.PTC.guard())
_r = PS.pt_op(dict(op="aim"))
check("명판 촬영 방식: 동의 없이는 조준 명령도 거부", _r[0] is False and "체크하지 않고" in (_r[1] or ""), _r)
world(5.8, 0.0, face, consent=True, new_scene=False); PS.poll_once()
check("명판 촬영 방식: 동의하고 시작한 실험이면 동의 때문에 막지 않음", PS.ST["exp"]["block"] is None and not any("팬틸트 카메라 사용" in w for w in PS.PTC.guard()), PS.PTC.guard())
(TMP / "O/runtime/plate_case1.on").unlink()
world(5.8, 0.0, face, consent=True, new_scene=False); PS.poll_once()
check("스위치가 꺼지면 예전 규칙: 동의로 시작한 실험은 거부", any("기존 자동 촬영" in w for w in PS.PTC.guard()), PS.PTC.guard())
world(5.8, 0.0, face, consent=False, new_scene=False); PS.poll_once()
check("스위치가 꺼지면 예전 규칙: 동의 없이 시작한 실험은 막지 않음", PS.ST["exp"]["block"] is None)
world(5.8, 0.0, face, new_scene=False)
_mt = PS.max_temp
PS.max_temp = lambda: 76.0
check("실험 중 본체 온도 76도면 온도로는 연결을 막지 않음", not any("온도" in w for w in PS.PTC.guard()), PS.PTC.guard())
PS.max_temp = lambda: 79.0
check("본체 온도 79도여도 온도로는 연결을 막지 않음", not any("온도" in w for w in PS.PTC.guard()), PS.PTC.guard())
PS.max_temp = lambda: 80.0
check("본체 온도 80도면 연결 거부 사유에 온도", any("온도 80" in w for w in PS.PTC.guard()), PS.PTC.guard())
PS.max_temp = _mt
_f = PS.finite(dict(rs=dict(thr=float("inf"), regions=[dict(sim=float("nan"), z=1.5)]), a=(1.0, float("-inf"))))
check("화면으로 보내는 값에 NaN·Infinity 가 없음(브라우저가 읽을 수 있는 JSON)", _f == dict(rs=dict(thr=None, regions=[dict(sim=None, z=1.5)]), a=[1.0, None])
      and "Infinity" not in json.dumps(_f, allow_nan=False), _f)
print("[6] 실험이 바뀔 때 · 정지 요청")
PS.ST.update(target=None, done=[], cand=None, auto_scene=None, case=1, auto=True, photo=None); PS.ST["e5req_seen"] = set()


def stop_req(oid, u, v, h_deg, tu, tv, up, age=1.5, with_pose=True):
    """E5 정지 요청 한 건(운영판 형식): age 초 전 장면에서, 로봇이 (u, v, h) 에 있을 때 본 표지 (tu, tv, up)"""
    now = time.time(); reg = plate_region(u, v, h_deg, tu, tv, up)
    x_, y_, yaw_ = G.planar_to_raw(u, v, math.radians(h_deg))
    r = dict(observation_id=oid, wall=now, detection_timestamp=now - age, bbox=reg["box"], depth_m=reg["z"], sim=0.13)
    if with_pose:
        r["odom_pose_at_detection"] = dict(status="INTERPOLATED", x=x_, y=y_, yaw=yaw_)
    return r


world(5.0, 0.0, 0.0, exp_id="X_A", requests=[stop_req("runA_O001", 5.0, 0.0, 0.0, 9.0, 1.1, UP)])
_t = PS.ST["target"]
check("정지 요청으로 표지를 기억(어느 실험에서 기억했는지 남김)", _t is not None and _t["src"] == "E5 정지 요청" and _t.get("exp") == "X_A" and abs(_t["uv"][0] - 9.0) < 0.2 and abs(_t["uv"][1] - 1.1) < 0.2, _t)
PS.ST["done"].append([3.0, 0.5])
world(5.0, 0.0, 0.0, exp_id="X_B", requests=[])               # 새 실험: FAST-LIO 가 새로 시작해 위치의 기준이 달라진다
check("실험이 바뀌면 앞 실험의 표지·끝낸 자리를 버림", PS.ST["target"] is None and PS.ST["done"] == [] and PS.ST["photo"] is None, PS.ST["target"])
world(2.0, 0.0, 0.0, exp_id="X_B", requests=[stop_req("runB_O001", 2.0, 0.0, 0.0, 6.0, -1.0, UP)])
_t = PS.ST["target"]
check("새 실험의 정지 요청은 새로 기억", _t is not None and _t.get("exp") == "X_B" and abs(_t["uv"][0] - 6.0) < 0.2 and abs(_t["uv"][1] + 1.0) < 0.2, _t)
PS.ST["target"]["exp"] = "X_OLD"; _a = PS.op_aim(); PS.ST["target"]["exp"] = "X_B"
check("다른 실험에서 기억한 표지로는 조준하지 않음", _a[0] is False and "앞 실험" in (_a[1] or ""), _a)
PS.ST.update(target=None)
world(2.0, 0.0, 0.0, exp_id="X_B", requests=[stop_req("runB_O002", 2.0, 0.0, 0.0, 7.0, 1.2, UP, age=6.0)])
check("장면이 6초 전이어도 요청에 든 그 장면의 위치로 기억", PS.ST["target"] is not None and abs(PS.ST["target"]["uv"][0] - 7.0) < 0.2, PS.ST["target"])
PS.ST.update(target=None)
world(2.0, 0.0, 0.0, exp_id="X_B", requests=[stop_req("runB_O003", 2.0, 0.0, 0.0, 7.0, 1.2, UP, age=6.0, with_pose=False)])
check("위치를 알 수 없는 요청은 기억하지 않고 이유를 기록에 남김", PS.ST["target"] is None and any("기억하지 못함" in l for l in PS.ST["log"][-3:]), PS.ST["log"][-3:])
world(2.0, 0.0, 0.0, exp_id="X_B", requests=[stop_req("runB_O004", 2.0, 0.0, 0.0, 6.0, 1.2, UP)])
_old = PS.ST["target"]["id"] if PS.ST["target"] else None
world(11.0, 0.0, 0.0, exp_id="X_B", requests=[])               # 끝내기를 누르지 않고 표지(6.0)를 5 m 지나침
check("표지를 지나치면 안내가 '지나쳤습니다'", _old is not None and PS.ST["guide"]["phase"] == "passed", PS.ST["guide"])
world(11.0, 0.0, 0.0, exp_id="X_B", requests=[])
check("새 정지 요청이 없으면 지나친 표지를 그대로 둠", PS.ST["target"] is not None and PS.ST["target"]["id"] == _old)
world(11.0, 0.0, 0.0, exp_id="X_B", requests=[stop_req("runB_O005", 11.0, 0.0, 0.0, 15.0, -1.2, UP)])
_t = PS.ST["target"]
check("지나친 표지는 비우고 새 정지 요청의 표지를 기억", _t is not None and _t["id"] != _old and abs(_t["uv"][0] - 15.0) < 0.2 and abs(_t["uv"][1] + 1.2) < 0.2, _t)
world(5.8, 0.0, 0.0, exp_id="X_TEST", requests=[])

print("[7] 표지 위치를 운영판 저장소 좌표로 옮겨 적기")
# ---- 표지 위치를 운영판 저장소(odom_link.csv)의 좌표로 옮겨 적기: 저장소 자세가 FAST-LIO 원래 값과 다를 때(표본 자세 보정)
(TMP / "store").mkdir(exist_ok=True)


def wstore(rows, tail=""):
    (TMP / "store/odom_link.csv").write_text("sample_id,t_sec,fastlio_x,fastlio_y,fastlio_yaw_rad,delta_s_m,delta_yaw_deg\n" +
                                             "".join("LIVE_%06d,%.6f,%.6f,%.6f,%.6f,0.0000,0.000\n" % ((i,) + tuple(r)) for i, r in enumerate(rows)) + tail)


_hist = [(1000.0 + 0.1 * i, 2.0, 3.0, 0.5) for i in range(41)]            # 멈춰 선 로봇의 FAST-LIO 원래 값 (t, x, y, yaw), 1000~1004 초
_r = (-2.0, 3.0, -0.5)                                                     # 같은 자세의 저장소 관례 (-x, y, -yaw)
(TMP / "store/odom_link.csv").unlink(missing_ok=True)
check("저장소 파일이 없으면 변환 없음", PS.store_frame(_hist) == dict(ok=False, why="NO_STORE"))
wstore([(1002.0,) + _r, (1003.5,) + _r])
_sf = PS.store_frame(_hist)
_p = PS.to_store(-1.0, 3.5, _sf)
check("저장소 자세가 원래 값 그대로면 표지 위치가 바뀌지 않음", _sf["ok"] and _sf["shift_m"] == 0.0 and _sf["dyaw_deg"] == 0.0 and abs(_sf["age_s"] - 0.5) < 1e-6
      and math.hypot(_p[0] + 1.0, _p[1] - 3.5) < 1e-9, _sf)
_a = math.radians(3.0); _c = (_r[0] + 0.30, _r[1] - 0.25, _r[2] + _a)       # 저장소가 39 cm, 3도 틀어진 상태
wstore([(1002.0,) + _r, (1003.5,) + _c])
_sf = PS.store_frame(_hist)
_p = PS.to_store(_r[0] + 1.0, _r[1] + 0.5, _sf)                            # 로봇에서 (1.0, 0.5) 떨어진 표지
_want = (_c[0] + math.cos(_a) * 1.0 - math.sin(_a) * 0.5, _c[1] + math.sin(_a) * 1.0 + math.cos(_a) * 0.5)
check("저장소 자세가 틀어져 있으면 로봇 기준 상대 위치를 지킨 채 저장소 좌표로 옮김", _sf["ok"] and abs(_sf["shift_m"] - 0.391) < 1e-3 and abs(_sf["dyaw_deg"] - 3.0) < 1e-2
      and math.hypot(_p[0] - _want[0], _p[1] - _want[1]) < 1e-6, (_sf, _p))
_q = PS.to_store(_r[0], _r[1], _sf)
check("로봇 자리 자체는 저장소의 로봇 자리로 간다", math.hypot(_q[0] - _c[0], _q[1] - _c[1]) < 1e-9, _q)
wstore([(1002.0,) + _r, (1003.5,) + _c], tail="LIVE_000002,1003.9,-9.0,9.0")
check("쓰는 중인 마지막 줄(줄바꿈 없음)은 쓰지 않음", PS.store_frame(_hist).get("row_t") == 1003.5)
wstore([(990.0,) + _c])
_sf = PS.store_frame(_hist)
check("저장소의 마지막 줄이 6초 넘게 오래됐으면 옮기지 않음", _sf["ok"] is False and _sf["why"] == "STORE_ROW_STALE", _sf)
wstore([(1003.5, _r[0] + 6.0, _r[1], _r[2])])
_sf = PS.store_frame(_hist)
check("차이가 터무니없이 크면(5 m 초과) 옮기지 않음", _sf["ok"] is False and _sf["why"] == "IMPLAUSIBLE", _sf)
wstore([(1010.0,) + _c, (1003.5,) + _c][::-1])
check("위치 기록보다 앞선 시각의 줄은 건너뛰고 그 앞 줄을 씀", PS.store_frame(_hist).get("row_t") == 1003.5)
wstore([(1002.0,) + _r, (1003.5,) + _c])
_tg = dict(uv=[4.0, 1.2], up=1.7, id=7, n_refine=1)
_ph = dict(name="001_c1_aim.jpg", wall=2000.0, stopped_wall=1999.0, stop=dict(t=1004.0, x=2.0, y=3.0, yaw=0.5), sign=dict(state="CENTERED", roi=[1, 2, 3, 4], sharp=50.0),
           pan=100.0, tilt=150.0, basis="기억한 위치", capture_id="c", store=PS.store_frame(_hist))
_rc = PS.make_record("plate_X_P001", "X", _ph, _tg, "/x.jpg", "0" * 64, [4056, 3040])
_xr, _yr = G.odom_link_point(4.0, 1.2)
_xs, _ys = PS.to_store(_xr, _yr, _ph["store"])
check("넘기는 줄: 표지 위치는 저장소 좌표, 원래 값은 따로, 촬영 자리는 FAST-LIO 원래 값 그대로",
      abs(_rc["target_odom_link_xy"][0] - _xs) < 2e-3 and abs(_rc["target_odom_link_xy"][1] - _ys) < 2e-3
      and abs(_rc["target_odom_link_xy_fastlio"][0] - _xr) < 2e-3 and abs(_rc["target_odom_link_xy_fastlio"][1] - _yr) < 2e-3
      and math.hypot(_xs - _xr, _ys - _yr) > 0.2 and _rc["odom_pose_at_stop"] == dict(t=1004.0, x=2.0, y=3.0, yaw=0.5)
      and _rc["store_frame"]["applied"] is True and abs(_rc["store_frame"]["shift_m"] - 0.391) < 1e-3, (_rc["target_odom_link_xy"], _rc["target_odom_link_xy_fastlio"], _rc["store_frame"]))
json.dumps(PS.finite(_rc), allow_nan=False)
# 실제 흐름: 허브 위치 기록이 여러 번 읽혀 이어 붙고, 촬영 순간의 변환이 사진 기록에 남는다
PS.ST.update(hist=[])
world(3.0, 0.0, 0.0, consent=True, exp_id="X_STORE", new_scene=False); time.sleep(0.25); world(3.0, 0.0, 0.0, consent=True, exp_id="X_STORE", new_scene=False)
_h = PS.ST["hist"]
check("위치 기록을 이어 붙여 둠(겹치는 시각은 한 번만, 시간순)", len(_h) > 31 and all(b[0] > a[0] for a, b in zip(_h, _h[1:])), len(_h))
_x, _y, _yaw = G.planar_to_raw(3.0, 0.0, 0.0)
wstore([(_h[-1][0] - 0.8, -_x + 0.2, _y, -_yaw)])
_sf = PS.store_frame(list(_h))
check("실제 형식의 허브 기록으로 변환이 구해짐(20 cm)", _sf["ok"] and abs(_sf["shift_m"] - 0.2) < 2e-3 and 0.7 < _sf["age_s"] < 0.9, _sf)
world(3.0, 0.0, 0.0, consent=True, exp_id="X_STORE2", new_scene=False)
check("실험이 바뀌면 이어 붙인 위치 기록도 새로 시작", len(PS.ST["hist"]) == 31, len(PS.ST["hist"]))
world(5.8, 0.0, 0.0, exp_id="X_TEST", requests=[])

print("모두 통과: %d개" % N["ok"])

if "--serve" in sys.argv:                      # 화면 확인용: 가짜 자료로 사이트를 18791 포트에 잠깐 띄운다 (장치 없음)
    import threading
    from http.server import ThreadingHTTPServer
    secs = float(sys.argv[sys.argv.index("--serve") + 1])
    PS.ST.update(case=1, auto=True, done=[], target=None, sel=None)
    PS.PTC.cmd = tuple(geo.home)
    srv = ThreadingHTTPServer(("0.0.0.0", 18791), PS.H); threading.Thread(target=srv.serve_forever, daemon=True).start()
    pos = TU - 1.02
    img = np.full((480, 640, 3), 70, np.uint8)
    for k in range(2):
        world(pos - 2.5, 0.0, 0.0, regions=[plate_region(pos - 2.5, 0.0, 0.0, TU, TV, UP), dict(box=[400.0, 250.0, 470.0, 330.0], sim=0.04, z=3.1, state="REJECT", display="REJECTED_POSTER")])
    t0 = time.time(); shot = False
    while time.time() - t0 < secs:
        far = time.time() - t0 < secs * 0.4
        u_now = pos - 2.5 if far else pos
        world(u_now, 0.0, 0.0, regions=[plate_region(u_now, 0.0, 0.0, TU, TV, UP)] if far else [], new_scene=True)
        if not far and not shot:
            shot = True; PS.op_aim(); wait_pt(); PS.op_nudge(3.0, -2.0); wait_pt()
        time.sleep(0.4)
    print("화면 확인용 서버 끝")
