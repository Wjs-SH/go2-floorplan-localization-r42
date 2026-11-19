#!/usr/bin/env python3
"""core_pin.py — 실험이 도는 동안 FAST-LIO 본체 스레드에 코어 하나(3번)를 전용으로 준다 (2026-11-13 시험, 11-15 부터 항상 적용).

파일은 하나도 고치지 않는다. 실행 중인 프로세스의 코어 지정(affinity)만 바꾸고, 실험이 끝나면 전부 원래대로 되돌린다.
  * FAST-LIO(fastlio_mapping)에서 가장 바쁜 스레드 -> 3번 코어만
  * 그 밖에 3번 코어를 쓸 수 있는 같은 사용자의 모든 스레드 -> 3번만 뺀다 (센서 허브는 2번 + 4번)
  * go2_control_runner(로봇 보행 제어)는 절대 건드리지 않는다. 다른 사용자·커널 스레드는 권한이 없어 그대로다.
동작 (기본값 = 항상 적용, 2026-11-15 사용자 지시 "영구 설정으로"):
  * 모든 실험에 적용한다(--first always). FAST-LIO 속도가 9.5 Hz 이상으로 안정되면 바꾸고, 안정될 때까지 계속 기다린다
  * 실험 도중에는 되돌리지 않는다(--rollback-hz 0). 속도가 낮아지면 기록에만 남긴다.
    근거: 11-13 18시 이후 되돌림 5번 중 3번은 FAST-LIO 가 비정상으로 시작한 실험, 1번은 실험이 끝나는 순간이었고,
    정상 실험 1번(X20261114_164041)은 되돌린 직후 30초 동안 8.5 Hz 미만 표본이 0 % -> 17 % 로 늘었다. 끝까지 유지한 27건은 10.0 Hz 였다
  * FAST-LIO 프로세스가 도중에 다시 뜨면 다시 적용한다
  * 실험 종료·신호(TERM/INT)·오류 때 원래대로 되돌린다(프로세스가 끝나면 지정도 사라지므로 다음 실험에서 다시 적용한다). 바꾸기 전 값은 backup.json 에도 남긴다
  * 실행 줄의 taskset 으로는 같은 배치를 만들 수 없다: 프로세스 전체가 한 코어에 묶여 FAST-LIO 의 보조 스레드까지 3번에 몰린다. 그래서 스레드 단위로 바꾼다
예전 동작(도중 되돌림): --rollback-hz 8.5 를 주면 바꾼 뒤 FAST-LIO 자체 처리 속도가 그 아래로 내려갈 때 되돌리고 그 실험에서는 다시 바꾸지 않는다.
비교 시험용(--first pin|baseline): 대기 시작 뒤 1번째 실험 = 전용 코어, 2번째 = 그대로(비교용), 3번째 = 전용 코어 ...
되돌림 판정은 FAST-LIO 가 스스로 처리한 스캔 속도(자체 기록 mat_pre.txt)로 한다. 수신 속도(lio_series 의 hz)는 몸체 IMU 브리지의 게이트가
출력을 1~5초 막을 때도 떨어지는데(11-13 되돌림 3번이 모두 이 경우였고 되돌린 뒤 오히려 느려졌다), 그것은 전용 코어 탓이 아니므로 되돌리지 않는다.

  python3 core_pin.py --wait-experiment [--arm-s 21600] [--first always|pin|baseline]
  python3 core_pin.py --rehearse 6        실험 없이: 가짜 바쁜 프로세스를 대상으로 바꿨다가 되돌리고, 모든 스레드가 원래 값인지 검사
  python3 core_pin.py --restore           backup.json 으로 되돌리기 (비정상 종료 뒤)
로그: <out>/core_pin.log"""
import argparse, json, os, re, signal, subprocess, sys, time
from pathlib import Path

OPR = Path.home() / "sci_ws/repairs/livefield2_20261018_151606"
EXP = OPR / "runtime/ctl/experiment.json"
LIO = OPR / "runtime/state/lio_series.jsonl"
ACTIVE = ("RUNNING", "PARTIAL", "PAUSED", "STARTING", "STOPPING")
CORE = 3
NEVER = re.compile(r"go2_control_runner")
HUB = re.compile(r"sensor_hub\.py")
TCK = os.sysconf("SC_CLK_TCK"); UID = os.getuid(); NCPU = os.cpu_count() or 8


def exp_state(path=EXP):
    try:
        d = json.loads(Path(path).read_text())
        return d.get("state"), d.get("exp_id")
    except (OSError, ValueError):
        return None, None


def cmdline(pid):
    try:
        return open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
    except OSError:
        return ""


def my_pids():
    out = []
    for d in os.listdir("/proc"):
        if d.isdigit() and int(d) != os.getpid():
            try:
                if os.stat(f"/proc/{d}").st_uid == UID:
                    out.append(int(d))
            except OSError:
                pass
    return out


def tids(pid):
    try:
        return [int(t) for t in os.listdir(f"/proc/{pid}/task")]
    except OSError:
        return []


def ticks(pid, tid):
    try:
        s = open(f"/proc/{pid}/task/{tid}/stat").read()
        f = s[s.rindex(")") + 2:].split()
        return int(f[11]) + int(f[12])
    except (OSError, ValueError, IndexError):
        return None


def last_hz(path=LIO):
    """(wall, hz) of the newest FAST-LIO rate sample, or (None, None)"""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2); n = f.tell(); f.seek(max(0, n - 6000))
            lines = [l for l in f.read().decode("utf-8", "replace").splitlines() if l.strip().startswith("{")]
        for l in reversed(lines):
            try:
                d = json.loads(l)
                return d.get("wall"), d.get("hz")
            except ValueError:
                continue
    except OSError:
        pass
    return None, None


MATPRE = OPR / "runtime/fastlio/Log/mat_pre.txt"


class ScanRate:
    """FAST-LIO 가 스스로 처리한 스캔 속도(Hz): 자체 기록 mat_pre.txt 에 스캔마다 한 줄이 붙는다. 최근 10초의 줄 수로 잰다.
    어댑터가 출력을 막아도(브리지 게이트) 이 값은 그대로다. 아직 잴 수 없으면 None."""

    def __init__(self, path=MATPRE, clock=time.time):
        self.path, self.clock, self.off, self.hist = path, clock, None, []

    def poll(self):
        try:
            n = os.path.getsize(self.path)
            if self.off is None or n < self.off:                  # 처음이거나 파일이 새로 시작됨
                self.off, self.hist = n, []
                return None
            with open(self.path, "rb") as f:
                f.seek(self.off); new = f.read(n - self.off).count(b"\n")
            self.off = n
        except OSError:
            return None
        now = self.clock()
        self.hist = [h for h in self.hist + [(now, new)] if now - h[0] <= 10.0]
        span = self.hist[-1][0] - self.hist[0][0]
        return (sum(c for _, c in self.hist[1:]) / span) if (len(self.hist) >= 3 and span >= 3.5) else None


def guard_reason(now, w, hz, own, rollback):
    """-> (되돌릴 이유 또는 None, 알릴 말 또는 None). own = FAST-LIO 자체 처리 속도(없으면 None), (w, hz) = 수신 속도 기록."""
    rx_dead = hz is None or not w or now - w > 8
    rx_low = (not rx_dead) and hz < rollback
    if own is None:                                               # 자체 기록을 못 읽으면 예전 규칙(수신 속도)
        return ("속도 기록이 멎음" if rx_dead else (f"속도 {hz} Hz < {rollback}" if rx_low else None)), None
    if own < rollback:
        return f"FAST-LIO 자체 처리 속도 {own:.1f} Hz < {rollback}", None
    if rx_dead or rx_low:
        return None, f"수신 속도만 낮음({'기록 멎음' if rx_dead else str(hz) + ' Hz'}), FAST-LIO 자체는 {own:.1f} Hz — 어댑터 게이트 끊김으로 보고 되돌리지 않음"
    return None, None


def find_fastlio():
    for p in my_pids():
        c = cmdline(p).split()
        if c and os.path.basename(c[0]) == "fastlio_mapping":
            return p
    return None


def hot_thread(pid, dt=2.0):
    a = {t: ticks(pid, t) for t in tids(pid)}
    time.sleep(dt)
    best, pct = None, 0.0
    for t in tids(pid):
        b = ticks(pid, t)
        if b is not None and a.get(t) is not None:
            v = 100.0 * (b - a[t]) / TCK / dt
            if v > pct:
                best, pct = t, v
    return best, pct


class Pin:
    def __init__(self, out, say):
        self.out, self.say = out, say
        self.orig, self.proc_orig, self.flags = {}, {}, {}      # tid -> (pid, mask) ; pid -> mask ; pid -> (never, hub)
        self.hot = None

    def _flag(self, pid):
        if pid not in self.flags:
            c = cmdline(pid)
            exe = os.path.basename(c.split()[0]) if c.split() else ""      # 2026-11-13: 실행 파일 이름으로만 판별한다. 명령줄 전체로 보면
            self.flags[pid] = (bool(NEVER.search(exe)), bool(HUB.search(c)))  # "--protect go2_control_runner" 를 가진 감독기 4개가 제외됐었다
        return self.flags[pid]

    def _set(self, pid, tid, mask):
        cur = sorted(os.sched_getaffinity(tid))
        if tid not in self.orig:
            self.orig[tid] = (pid, cur)
            self.proc_orig.setdefault(pid, cur)
        os.sched_setaffinity(tid, mask)

    def pin_hot(self, pid, tid):
        self.hot = tid
        self._set(pid, tid, {CORE})

    def sweep(self):
        """move every other thread of this user off CORE; -> (changed, errors)"""
        n = e = 0
        for pid in my_pids():
            never, hub = self._flag(pid)
            if never:
                continue
            for tid in tids(pid):
                if tid == self.hot:
                    continue
                try:
                    cur = os.sched_getaffinity(tid)
                    if CORE not in cur:
                        continue
                    new = set(cur) - {CORE}
                    if hub:
                        new |= {4}
                    if not new:
                        continue
                    self._set(pid, tid, new); n += 1
                except OSError:
                    e += 1
        if n:
            self.save()
        return n, e

    def save(self):
        tmp = self.out / "backup.json.tmp"
        tmp.write_text(json.dumps(dict(wall=time.time(), hot=self.hot, proc_orig={str(k): v for k, v in self.proc_orig.items()},
                                       orig={str(k): v for k, v in self.orig.items()})))
        os.replace(tmp, self.out / "backup.json")

    def restore(self):
        n = e = 0
        for pid, pmask in list(self.proc_orig.items()):
            for tid in tids(pid):
                mask = self.orig[tid][1] if tid in self.orig else pmask       # threads born after the change inherited the reduced mask
                try:
                    if sorted(os.sched_getaffinity(tid)) != sorted(mask):
                        os.sched_setaffinity(tid, mask); n += 1
                except OSError:
                    e += 1
        self.orig, self.proc_orig, self.flags, self.hot = {}, {}, {}, None
        try:
            (self.out / "backup.json").unlink()
        except OSError:
            pass
        return n, e


def restore_from_backup(out, say):
    p = out / "backup.json"
    if not p.exists():
        say("backup.json 없음: 되돌릴 것이 없다"); return
    d = json.loads(p.read_text()); n = 0
    for pid, pmask in d["proc_orig"].items():
        for tid in tids(int(pid)):
            mask = d["orig"].get(str(tid), [None, pmask])[1]
            try:
                os.sched_setaffinity(tid, mask); n += 1
            except OSError:
                pass
    p.unlink(); say(f"backup.json 으로 {n}개 스레드를 되돌림")


def snapshot():
    out = {}
    for pid in my_pids():
        for tid in tids(pid):
            try:
                out[tid] = sorted(os.sched_getaffinity(tid))
            except OSError:
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-experiment", action="store_true"); ap.add_argument("--arm-s", type=float, default=21600.0)
    ap.add_argument("--first", choices=["always", "pin", "baseline"], default="always"); ap.add_argument("--rollback-hz", type=float, default=0.0)
    ap.add_argument("--rehearse", type=float, default=0.0); ap.add_argument("--restore", action="store_true")
    ap.add_argument("--out", default=str(Path.home() / "sci_ws_student_staging_20261109/_runs/core_pin"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log = open(out / "core_pin.log", "a")

    def say(m):
        log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {m}\n"); log.flush()
        if a.rehearse or a.restore:
            print(m)

    if a.restore:
        restore_from_backup(out, say); return
    os.nice(15)
    try:
        os.sched_setaffinity(0, set(range(NCPU)) - {CORE})
    except OSError:
        pass
    P = Pin(out, say)

    def bye(sig, frm):
        n, e = P.restore(); say(f"신호 {sig}: {n}개 스레드 되돌림 (실패 {e}), 종료"); sys.exit(0)
    signal.signal(signal.SIGTERM, bye); signal.signal(signal.SIGINT, bye)

    if a.rehearse:
        before = snapshot()
        child = subprocess.Popen([sys.executable, "-c", "import time\nt=time.time()\nwhile time.time()-t<%f: pass" % (a.rehearse + 6)])
        try:
            time.sleep(1.0)
            hot, pct = hot_thread(child.pid, 1.0)
            P.pin_hot(child.pid, hot or child.pid)
            n, e = P.sweep()
            say(f"예행: 가짜 대상 pid {child.pid} 스레드 {hot} ({pct:.0f} %) -> {CORE}번 전용, 다른 스레드 {n}개를 {CORE}번에서 뺌 (실패 {e})")
            mid = snapshot()
            on3 = [t for t, m in mid.items() if CORE in m and t != P.hot]
            never = [p for p in my_pids() if cmdline(p).split() and NEVER.search(os.path.basename(cmdline(p).split()[0]))]
            say(f"예행: 바꾼 뒤 {CORE}번을 쓸 수 있는 다른 스레드 {len(on3)}개 (보행 제어 프로세스 {never} 의 스레드 {sum(len(tids(p)) for p in never)}개는 건드리지 않음)")
            time.sleep(a.rehearse)
        finally:
            n, e = P.restore(); child.kill()
        after = snapshot()
        diff = [t for t in before if t in after and before[t] != after[t]]
        say(f"예행: {n}개 되돌림 (실패 {e}). 되돌린 뒤 원래 값과 다른 스레드 {len(diff)}개 (0 이어야 함)")
        return

    if not a.wait_experiment:
        ap.error("--wait-experiment | --rehearse S | --restore")
    keep = a.rollback_hz <= 0                    # 항상 유지: 도중에 되돌리지 않고, FAST-LIO 가 다시 뜨면 다시 적용
    say(f"대기 시작 pid {os.getpid()}, {a.arm_s:.0f} s, 첫 실험 = {a.first}, " + ("도중 되돌림 없음(항상 유지)" if keep else f"되돌림 기준 {a.rollback_hz} Hz"))
    t_arm, k = time.time(), 0
    while time.time() - t_arm < a.arm_s:
        st, xid = exp_state()
        if st not in ACTIVE:
            time.sleep(2.0); continue
        pinned = True if a.first == "always" else ((k % 2 == 0) == (a.first == "pin")); k += 1
        say(f"실험 {xid} {st}: {'전용 코어' if pinned else '그대로 둠(비교용)'}")
        done, rolled, t_apply, ok, t0, fl, guard = False, False, None, 0, time.time(), None, False
        SR, t_note = ScanRate(), 0.0
        n_apply, lost_t, waited, n_fail = 0, None, False, 0
        ended = 0
        while True:
            time.sleep(2.0)
            s2, _ = exp_state()
            ended = ended + 1 if (s2 is not None and s2 not in ACTIVE) else 0
            if ended >= 2:
                break
            if not pinned or rolled:
                continue
            w, hz = last_hz()
            if not done:
                fl = find_fastlio()
                ok = ok + 1 if (fl and hz is not None and w and time.time() - w < 5 and hz >= 9.5) else 0
                if ok >= 3:
                    try:
                        hot, pct = hot_thread(fl, 2.0)
                        if hot is None or pct < 20.0:
                            hot = fl
                        P.pin_hot(fl, hot)
                        n, e = P.sweep(); P.save()
                        done, t_apply = True, time.time()
                        n_apply += 1
                        say(f"  적용: FAST-LIO pid {fl} 스레드 {hot} ({pct:.0f} %) -> {CORE}번 전용, 다른 스레드 {n}개를 {CORE}번에서 뺌 (실패 {e}), 그때 속도 {hz} Hz")
                    except OSError as ex:
                        n, e = P.restore(); n_fail += 1
                        if keep and n_fail < 5:      # 항상 유지: 스레드가 그 사이 사라진 것 같은 일시 오류는 다시 시도한다
                            ok = 0; say(f"  적용 실패({ex}) -> {n}개 되돌림, 다시 시도 ({n_fail}/5)")
                        else:
                            rolled = True; say(f"  적용 실패({ex}) -> {n}개 되돌림, 이 실험은 손대지 않음")
                elif time.time() - t0 > 240:
                    if not keep:
                        rolled = True; say("  FAST-LIO 가 240 s 안에 9.5 Hz 로 안정되지 않음 -> 이 실험은 손대지 않음")
                    elif not waited:
                        waited = True; say("  FAST-LIO 가 240 s 안에 9.5 Hz 로 안정되지 않음 -> 안정될 때까지 계속 기다림")
                continue
            if not os.path.isdir(f"/proc/{fl}"):
                n, e = P.restore(); lost_t = time.time()
                if keep:                         # FAST-LIO 가 다시 뜨면 처음부터 다시 적용한다
                    done, ok, t0, waited, SR = False, 0, time.time(), False, ScanRate()
                    say(f"  FAST-LIO 프로세스가 사라짐 -> {n}개 되돌림 (다시 뜨면 다시 적용)")
                else:
                    rolled = True; say(f"  FAST-LIO 프로세스가 사라짐 -> {n}개 되돌림")
                continue
            n, e = P.sweep()
            if n:
                say(f"  새로 생긴 스레드 {n}개를 {CORE}번에서 뺌")
            own = SR.poll()
            if time.time() - t_apply > 12.0:
                bad, note = guard_reason(time.time(), w, hz, own, 8.5 if keep else a.rollback_hz)
                if keep and bad:                 # 항상 유지: 기록만 하고 되돌리지 않는다
                    bad, note = None, f"{bad} — 되돌리지 않음(항상 유지)"
                if note and time.time() - t_note > 30.0:
                    t_note = time.time(); say("  " + note)
                if bad:
                    n, e = P.restore(); rolled = guard = True
                    say(f"  되돌림 (+{time.time() - t_apply:.0f} s): {bad} -> {n}개 스레드 원래대로 (실패 {e}). 이 실험에서는 다시 바꾸지 않음")
        n, e = P.restore()
        if guard:
            how = " — 안전장치로 도중에 되돌렸음"
        elif done or (n_apply and lost_t is not None and time.time() - lost_t < 40.0):
            how = " — 끝까지 전용 코어 유지" + (f" (도중에 FAST-LIO 가 다시 떠서 {n_apply}번 적용)" if n_apply > 1 else "")
        elif n_apply:
            how = f" — FAST-LIO 가 실험이 끝나기 {time.time() - lost_t:.0f}초 전에 사라져 그때까지만 유지 (적용 {n_apply}번)"
        else:
            how = " — 적용하지 못함(FAST-LIO 가 9.5 Hz 로 안정된 적이 없음)" if pinned else ""
        say(f"실험 {xid} 끝: {n}개 스레드 되돌림 (실패 {e}){how}")
        if pinned and not done and a.first != "always":   # 11-13 07:12: 시작에 실패한 실험이 '전용 코어' 차례를 써 버려 본 실험이 비교용이 됐다
            k -= 1
            say("  전용 코어를 적용하지 못한 실험이라 차례를 넘기지 않음 -> 다음 실험도 전용 코어 시험")
    say("대기 시간 끝, 종료")


if __name__ == "__main__":
    main()
