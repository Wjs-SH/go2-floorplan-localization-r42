#!/usr/bin/env python3
"""e7_proc.py (2026-11-17, 사용자 지시 "E7 을 e34 와 다른 프로세스로 떼어 내줘, 백업은 꼭") — E7 구간 정합을 e34 밖의 프로세스에서 계산한다.

왜: E7 은 e34 안의 스레드였다. 파이썬은 한 프로세스에서 한 번에 한 스레드만 파이썬 코드를 돌리므로(GIL) E3·E8·SigLIP 요청과 번갈아 돌았다.
    11-16 현장: E7 한 번 56~118 s, 정합 1 회 ≈ 15 s(212712: 118 s / 8 회). 빈 코어에서는 정합 1 회 2.5 s(e6e7_link R41 주석), 노트북 0.3 s.
무엇을: e6e7_link._run 의 '이어 맞추기(R40) + E7.evaluate' 부분만. 계산식·순서·예외 처리는 _run 과 같다(아래 compute 는 그 코드를 그대로 옮긴 것).
    E2 경쟁 후보 사전 선별, 판정(global_decision·명판 규칙·FUSED_FLOOR)·기록은 e34 에 그대로 남는다.
어떻게: e7_long 의 작업 프로세스와 같은 방식 — taskset 0,1,6,7(E3 코어 5 는 쓰지 않음), nice 5(E7 스레드와 같은 값), 수치 연산 스레드 1 개, python -s -B.
    e34 의 E7 스레드가 작업 파일(pickle)을 쓰고 프로세스를 띄운 뒤 기다린다(기다리는 동안 GIL 을 쓰지 않음). 결과 파일에서 후보(이어 맞춘 변환)·E7 결과를 받고,
    이 프로세스가 걸러 낸 구조 점(_SF_CACHE)을 e34 의 E7 캐시에 넣어 뒤따르는 FUSED_FLOOR·창 증거가 같은 계산을 다시 하지 않게 한다.
실패하면: 프로세스가 결과 없이 끝나면 e34 스레드에서 예전처럼 계산한다(THREAD_FALLBACK). TIMEOUT_S 안에 안 끝나면 그 프로세스(자기 자식, PID)만 끝내고
    E7 결과를 MORE_OBSERVATION_REQUIRED 로 둔다(확정 안 함). e34 가 죽으면 이 프로세스도 1 초 안에 스스로 끝난다.
기록: <state>/e7_proc.jsonl (E7 한 번마다 한 줄: 방식, 걸린 시간, 정합 횟수), 오류 출력 <state>/e7_proc.err.
끄기: e34 환경변수 LIVE2_E7_PROC=0 (예전처럼 e34 스레드에서 계산)."""
import importlib.util
import json
import os
import pickle
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORES = "0,1,6,7"          # e7_long.LongStructure.cores() 와 같음
NICE = 5                   # e6e7_link._run 이 E7 스레드에 주던 값
TIMEOUT_S = 300.0


def enabled():
    return os.environ.get("LIVE2_E7_PROC", "1") == "1"


def compute(store, light, E7, cands, r, t_end, win_s):
    """e6e7_link._run (7cf0b8b6d49e) 의 이어 맞추기 + evaluate 를 그대로. cands 의 rec 는 제자리에서 바뀐다. -> res"""
    lights = {c["rec"]["light"]: light(c["rec"]["light"]) for c in cands}
    try:                                                                    # R40: the candidates E8 accepted get the same continuation from their search bound as the fair floor comparison (e7_long)
        import e7_long as EL_
        fr_ = E7.frames(store, t_end, 90.0, structural=True)
        for c in cands:
            if c["candidate_id"] in (r.get("consistent_ids") or []) and len(fr_) >= 2:
                for _ in range(2):
                    rg = E7.register(lights[c["rec"]["light"]], c["rec"], fr_)
                    if not rg.get("on_bound"):
                        break
                    c["rec"] = EL_.recentered(c["rec"], fr_, rg["correction"])
    except Exception:                                                       # noqa: BLE001  the evaluation below decides as before
        pass
    try:
        res = E7.evaluate(store, cands, lights, t_end, "E6_OPERATOR_ASSISTED" if r.get("source") == "OPERATOR_ASSISTED"
                          else "E6_SEMANTIC", window_s=win_s, structural=True, semantic_ok=True)
    except Exception as ex:                                                 # noqa: BLE001
        res = dict(final_decision="MORE_OBSERVATION_REQUIRED", why=f"E7 error {type(ex).__name__}: {ex}", results=[])
    return res


def _seed(E7, sf):
    """이 프로세스가 걸러 낸 구조 점을 e34 의 E7 캐시에 넣는다(같은 표본 → 같은 값). 캐시 크기 상한은 E7 과 같게."""
    cache, mx = getattr(E7, "_SF_CACHE", None), getattr(E7, "_SF_MAX", 400)
    if cache is None:
        return 0
    n = 0
    for k, v in (sf or {}).items():
        if k in cache:
            continue
        if len(cache) >= mx:
            cache.pop(next(iter(cache)))
        cache[k] = v
        n += 1
    return n


def _log(state_dir, rec):
    try:
        with open(Path(state_dir) / "e7_proc.jsonl", "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except OSError:
        pass


def run(state_dir, store, cands, r, t_end, win_s, E7, timeout_s=TIMEOUT_S, py=None):
    """e34 의 E7 스레드에서 부른다. -> (dict(cands, res) 또는 None, 기록). None 이면 부른 쪽이 예전처럼 스레드에서 계산한다."""
    t0 = time.time()
    d = Path(state_dir) / "e7_proc"
    d.mkdir(exist_ok=True)
    tag = f"{os.getpid()}_{int(t0 * 1000)}"
    job, out = d / f"job_{tag}.pkl", d / f"out_{tag}.pkl"
    rec = dict(kind="LIVE2_E7_PROC", observation_id=r.get("observation_id"), source=r.get("source"), wall=t0, n_cands=len(cands), cores=CORES)
    got, p = None, None
    try:
        tmp = d / f".job_{tag}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(dict(store=str(store.root), cands=cands, r={k: r.get(k) for k in ("source", "consistent_ids", "observation_id")},
                             t_end=t_end, win_s=win_s, e7_path=str(Path(E7.__file__).resolve()), out=str(out)), f, protocol=4)
        os.replace(tmp, job)
        with open(Path(state_dir) / "e7_proc.err", "ab") as err:
            p = subprocess.Popen(["taskset", "-c", CORES, py or sys.executable, "-s", "-B", str(Path(__file__).resolve()), str(job)],
                                 stdout=subprocess.DEVNULL, stderr=err, env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"))
            try:
                rc = p.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                p.kill()                                                    # our own child (Popen object), nothing else
                p.wait()
                rc = None
        rec.update(pid=p.pid, rc=rc)
        if rc is None:
            rec["mode"] = "PROCESS_TIMEOUT"
            got = dict(cands=cands, res=dict(final_decision="MORE_OBSERVATION_REQUIRED", why=f"E7 worker gave no result in {timeout_s:.0f} s", results=[]))
        elif rc == 0 and out.exists():
            with open(out, "rb") as f:
                w = pickle.load(f)
            got = dict(cands=w["cands"], res=w["res"])
            rec.update(mode="PROCESS", worker_s=w.get("sec"), n_register=w.get("n_register"), n_sf=len(w.get("sf") or {}), seeded=_seed(E7, w.get("sf")))
        else:
            rec.update(mode="THREAD_FALLBACK", error=f"worker exit {rc}, result {'present' if out.exists() else 'missing'}")
    except Exception as ex:                                                 # noqa: BLE001  the caller computes in its own thread
        got = None
        rec.update(mode="THREAD_FALLBACK", error=f"{type(ex).__name__}: {ex}")
        if p is not None and p.poll() is None:                              # our own child only
            p.kill()
            p.wait()
    finally:
        for f_ in (job, out):
            try:
                f_.unlink()
            except OSError:
                pass
    rec["wall_s"] = round(time.time() - t0, 2)
    _log(state_dir, rec)
    return got, rec


def worker(job_path):
    t0 = time.time()
    ppid = os.getppid()

    def watch():                                                            # e34 gone -> stop (no orphan computing for nobody)
        while True:
            time.sleep(1.0)
            if os.getppid() != ppid:
                os._exit(3)
    threading.Thread(target=watch, daemon=True).start()
    try:
        os.setpriority(os.PRIO_PROCESS, 0, max(NICE, os.getpriority(os.PRIO_PROCESS, 0)))
    except (AttributeError, OSError):
        pass
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE.parent / "external" / "sci_live" / "floor_id" / "localizer"))   # e7_long.worker 와 같음
    import e3_tracker as E3
    with open(job_path, "rb") as f:
        j = pickle.load(f)
    spec = importlib.util.spec_from_file_location("live2_e7_rematch", j["e7_path"])            # e6e7_link._load_e7 와 같은 파일
    E7 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E7)
    n = [0]
    reg = E7.register

    def counted(*a, **k):
        n[0] += 1
        return reg(*a, **k)
    E7.register = counted                                                   # count only (evaluate looks the name up in the module)
    store = E3.Store(j["store"])
    store.refresh()
    lc = {}

    def light(p):
        if p not in lc:
            lc[p] = E3.LightMap(p)
        return lc[p]
    cands = j["cands"]
    res = compute(store, light, E7, cands, j["r"], j["t_end"], j["win_s"])
    out = Path(j["out"])
    tmp = out.with_name("." + out.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(dict(cands=cands, res=res, sf=dict(E7._SF_CACHE), n_register=n[0], sec=round(time.time() - t0, 2)), f, protocol=4)
    os.replace(tmp, out)


if __name__ == "__main__":
    worker(sys.argv[1])
