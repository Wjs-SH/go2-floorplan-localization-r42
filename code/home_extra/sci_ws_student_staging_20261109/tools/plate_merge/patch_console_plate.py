#!/usr/bin/env python3
"""patch_console_plate.py — 운영 콘솔(candidate/live2/console.py)에 학생 병합용 덩어리 두 개를 끼워 넣는다 (표시 PLATE_MERGE 사이).

 (1) start_module 의 E6 갈래 맨 앞: "명판 촬영(케이스 1) 방식" 스위치.
     운영폴더 runtime/plate_case1.on 이 있으면, '팬틸트 카메라 사용'에 동의하고 시작한 실험에서도 E6 자동 촬영 모듈을 띄우지 않는다
     (동의 없이 시작했을 때와 같은 "OFF" 처리). 동의 기록(승인 B, experiment.json 의 pantilt_consent)은 그대로 남는다.
 (2) a_start 의 실험 시작 기록 바로 뒤: 학생 도구 시작.
     운영폴더 runtime/student_start_hook 에 스크립트 경로가 적혀 있으면 실험을 시작할 때 그 스크립트를 떼어서 돌린다(촬영 도구·전용 코어 도구).
     실패해도 실험 시작은 그대로 간다.
두 파일이 모두 없으면 원래 동작과 한 글자도 다르지 않다.
 (3) [선택, --other-account] DeviceGate.identify: 추가 Tailscale 계정 허용.
     운영폴더 runtime/extra_tailscale_accounts 에 적힌 계정(한 줄에 하나, # 은 설명)의 기기를 로봇 소유 계정의 기기와 똑같이 다룬다:
     보기는 허용, 조작은 허용 목록(runtime/ctl/auth/trusted_devices.json)에 있는 기기만. 파일이 없거나 비어 있으면 원래 동작과 같다.
     이 덩어리는 --other-account 를 줄 때만 넣고, 이미 들어 있는 파일에 다시 넣을 때는 유지한다. 빼려면 --no-other-account (또는 --revert).

  python3 patch_console_plate.py <console.py> --check
  python3 patch_console_plate.py <console.py> --out <새 파일>
  python3 patch_console_plate.py <console.py> --other-account --out <새 파일>      (3)번 덩어리도 넣는다
  python3 patch_console_plate.py <console.py> --no-other-account --out <새 파일>   (3)번 덩어리만 뺀다
  python3 patch_console_plate.py <console.py> --revert --out <새 파일>     (빼면 원본과 바이트까지 같아야 한다)
입력 파일은 직접 덮어쓰지 않는다. 콘솔은 파이썬 프로그램이라 바꾼 뒤 콘솔을 다시 띄워야 적용된다."""
import argparse, ast, hashlib, os, re

MARK_B, MARK_E = "# PLATE_MERGE_BEGIN", "# PLATE_MERGE_END"
A1 = ('    def start_module(self, job, m, building):\n'
      '        if m == "e6":\n'
      '            apb = C.approval_status("B", "pantilt")\n')
B1 = ('            ' + MARK_B + ' E6 = 명판 촬영 방식 (학생 병합, 2026-11-13; 11-15 사용자 결정 "지금 방식으로 E6 을 대체"). runtime/plate_case1.on 이 있으면\n'
      '            # 동의하고 시작한 실험의 E6 는 명판 촬영 방식으로 한다: 예전 자동 촬영 프로그램(e6_precision_bridge)은 띄우지 않고, 팬틸트는 운용 탭의\n'
      '            # 명판 촬영 카드(촬영 도구 8791)가 쓴다. 승인 B 기록은 그대로 남는다. 모듈 기록의 plate=True 와 사유 문구로 이 방식임을 남긴다.\n'
      '            if apb["valid"] and (L / "runtime" / "plate_case1.on").exists():\n'
      '                why = "E6 = 명판 촬영 방식 — 운용 탭 카드에서 조준·촬영 (예전 자동 촬영 프로그램은 실행하지 않음)"\n'
      '                self.exp.mod(m, state="OFF", why=[why], plate=True)\n'
      '                self.step(job, module=m, ok=True, why=why)\n'
      '                self.exp.event("e6_plate", mode="CASE1")\n'
      '                return True\n'
      '            ' + MARK_E + '\n')
A2 = ('        X.event("experiment_start", building=building, siglip=siglip, input=input_mode, e6_mode=e6_mode, loc_mode=loc_mode, archived_leftover_files=moved,\n'
      '                start_gt=bool(sg))\n')
B2 = ('        ' + MARK_B + ' 학생 도구 시작 (학생 병합, 2026-11-13). runtime/student_start_hook 에 스크립트 경로가 적혀 있으면 실험을 시작할 때\n'
      '        # 그 스크립트를 떼어서 돌린다(촬영 도구·전용 코어 도구). 파일이 없으면 아무 일도 하지 않는다. 실패해도 실험 시작은 그대로 간다.\n'
      '        try:\n'
      '            import subprocess as _sp\n'
      '            _hook = (L / "runtime" / "student_start_hook").read_text().strip()\n'
      '            if _hook and Path(_hook).is_file():\n'
      '                _sp.Popen(["/bin/bash", _hook], stdin=_sp.DEVNULL, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL, start_new_session=True, close_fds=True)\n'
      '                X.event("student_start_hook", script=_hook)\n'
      '        except (OSError, ValueError):\n'
      '            pass\n'
      '        ' + MARK_E + '\n')
A3 = ('            same = bool(self.owner) and w["user"] == self.owner\n'
      '            trust = w["stable_id"] in self.trusted()\n')
TAG3 = "추가 Tailscale 계정 허용"
B3 = ('            ' + MARK_B + ' ' + TAG3 + ' (학생 병합, 2026-11-15). 운영폴더 runtime/extra_tailscale_accounts 에 적힌 계정(한 줄에 하나)의\n'
      '            # 기기는 로봇 소유 계정의 기기와 똑같이 다룬다: 보기는 허용, 조작은 허용 목록(trusted_devices.json)에 있는 기기만.\n'
      '            # 파일이 없거나 비어 있으면 원래 동작과 같다.\n'
      '            if not same and w["user"]:\n'
      '                try:\n'
      '                    _acc = {_l.strip().lower() for _l in (L / "runtime" / "extra_tailscale_accounts").read_text().splitlines()\n'
      '                            if _l.strip() and not _l.lstrip().startswith("#")}\n'
      '                except (OSError, ValueError):\n'
      '                    _acc = set()\n'
      '                same = str(w["user"]).strip().lower() in _acc\n'
      '            ' + MARK_E + '\n')
STRIP = re.compile(r"[ \t]*" + re.escape(MARK_B) + r".*?" + re.escape(MARK_E) + r"\n", re.S)
B = MARK_B


def strip(s):
    return STRIP.sub("", s)


def apply(s, other=None):
    """other: True = (3)번도 넣는다, False = (3)번은 뺀다, None = 들어 있으면 유지"""
    if other is None:
        other = (MARK_B + " " + TAG3) in s
    s = strip(s)
    for name, a in (("start_module 의 E6 갈래", A1), ("a_start 의 실험 시작 기록", A2)) + ((("DeviceGate.identify 의 계정·허용 목록 확인", A3),) if other else ()):
        if s.count(a) != 1:
            raise SystemExit("중단: %s 를 찾을 수 없거나 여러 곳입니다 (%d곳) — console.py 가 예상과 다릅니다" % (name, s.count(a)))
    for must in ('L = C.LROOT', 'self.exp.event("e6_off"', 'def step(self, job, **k):', 'from pathlib import Path'):
        if must not in s:
            raise SystemExit("중단: console.py 에 %r 가 없습니다" % must)
    s = s.replace(A1, A1 + B1).replace(A2, A2 + B2)
    if other:
        if 'control=same and trust' not in s or 'read=same' not in s:
            raise SystemExit("중단: DeviceGate.identify 의 결과 줄이 예상과 다릅니다")
        s = s.replace(A3, A3 + B3)
    ast.parse(s)                                          # 문법이 깨지면 여기서 멈춘다
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("--out"); ap.add_argument("--check", action="store_true"); ap.add_argument("--revert", action="store_true")
    ap.add_argument("--other-account", action="store_true"); ap.add_argument("--no-other-account", action="store_true")
    a = ap.parse_args()
    raw = open(a.src, "rb").read(); s = raw.decode("utf-8")
    new = strip(s) if a.revert else apply(s, True if a.other_account else (False if a.no_other_account else None))
    nb = new.encode("utf-8")
    assert strip(new) == strip(s)
    sha = lambda b: hashlib.sha256(b).hexdigest()[:16]
    print("입력  %s  sha256 %s  %d바이트%s" % (a.src, sha(raw), len(raw), "  (이미 끼워져 있음)" if B in s else ""))
    print("결과  sha256 %s  %d바이트  (%+d)%s" % (sha(nb), len(nb), len(nb) - len(raw), "  [추가 계정 허용 덩어리 포함]" if (MARK_B + " " + TAG3) in new else ""))
    if a.check:
        print("점검만 했습니다 — 아무것도 쓰지 않았습니다"); return
    if not a.out or os.path.realpath(a.out) == os.path.realpath(a.src):
        raise SystemExit("--out 이 필요합니다 (입력 파일은 직접 덮어쓰지 않습니다)")
    open(a.out, "wb").write(nb); print("씀   ", a.out)


if __name__ == "__main__":
    main()
