"""auth.py — LIVE2 operator accounts and persistent device sessions (stdlib only).

  account   : one operator id ("recs"). The password is chosen by the operator at first enrollment; there is no default
              password. Stored as scrypt(n=2^15, r=8, p=1, 32-byte salt) — never in plain text.
  enrollment: needs the robot-side setup code (runtime/ctl/operator_token, owner-only, made by `ctl console-token`), so a
              stranger who reaches the page first cannot claim the account. Password reset = robot-side CLI only.
  sessions  : 32-byte CSPRNG ids; the server keeps only sha256(id). "remember" = 30 days, otherwise 12 h with a browser
              session cookie. Stored on disk (0600) so a console restart or a reboot keeps the tablet logged in.
              A password change or "log out everywhere" revokes every session. CSRF token = HMAC(server secret, session id).
  approval  : NOT here. A valid login never implies approval A (boot-bound, time-bound, per-module) — ctl checks that.
"""
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from pathlib import Path

USER = "recs"
REMEMBER_S = 30 * 86400
SESSION_S = 12 * 3600
SCRYPT = dict(n=2 ** 15, r=8, p=1)
MAXMEM = 64 * 1024 * 1024
IP_FAILS, IP_LOCK_S = 5, 60
USER_FAILS, USER_LOCK_S = 20, 900


def _write_private(path, obj):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _read(path, default):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def _scrypt(pw, salt, n, r, p):
    return hashlib.scrypt(pw.encode(), salt=bytes.fromhex(salt), n=n, r=r, p=p, maxmem=MAXMEM, dklen=32).hex()


def password_problem(user, pw):
    """Strong enough: >= 10 chars and 3 of {lower, upper, digit, symbol}, or >= 16 chars; never contains the id."""
    if not isinstance(pw, str) or len(pw) < 10:
        return "비밀번호는 10자 이상이어야 합니다"
    if len(pw) > 200:
        return "비밀번호가 너무 깁니다"
    if user.lower() in pw.lower():
        return "비밀번호에 아이디를 넣을 수 없습니다"
    classes = sum(any(f(c) for c in pw) for f in (str.islower, str.isupper, str.isdigit, lambda c: not c.isalnum()))
    if len(pw) < 16 and classes < 3:
        return "영문 대·소문자, 숫자, 기호 중 3종류 이상을 섞거나 16자 이상으로 하십시오"
    return None


class AuthStore:
    def __init__(self, root, setup_code_file):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)
        self.setup_code_file = Path(setup_code_file)
        self.lock = threading.RLock()
        self.fails_ip, self.fails_user = {}, {}
        sec = self.root / "secret"
        if not sec.exists():
            _write_private(sec, dict(secret=secrets.token_hex(32), created=time.time()))
        self.secret = bytes.fromhex(_read(sec, {})["secret"])
        self._seen_write = {}

    # ---------------- files
    @property
    def accounts(self):
        return _read(self.root / "accounts.json", dict(version=1, users={}))

    def _save_accounts(self, d):
        if (self.root / "accounts.json").exists():                     # one rollback copy of the previous account file
            _write_private(self.root / "accounts.prev.json", self.accounts)
        _write_private(self.root / "accounts.json", d)

    def _sessions(self):
        return _read(self.root / "sessions.json", dict(version=1, sessions={}))

    def _save_sessions(self, d):
        _write_private(self.root / "sessions.json", d)

    @staticmethod
    def _h(sid):
        return hashlib.sha256(sid.encode()).hexdigest()

    # ---------------- enrollment / password
    def needs_enrollment(self, user=USER):
        return user not in self.accounts["users"]

    def setup_code_ok(self, code):
        try:
            real = self.setup_code_file.read_text().strip()
        except OSError:
            return False
        return bool(real) and isinstance(code, str) and hmac.compare_digest(code.encode(), real.encode())

    def enroll(self, user, password, setup_code):
        with self.lock:
            if user != USER:
                return "UNKNOWN_USER"
            if not self.needs_enrollment(user):
                return "ALREADY_ENROLLED"
            if not self.setup_code_ok(setup_code):
                return "SETUP_CODE_WRONG"
            p = password_problem(user, password)
            if p:
                return "WEAK_PASSWORD: " + p
            d = self.accounts
            d["users"][user] = self._hash_record(password, pw_version=1)
            self._save_accounts(d)
            return None

    def _hash_record(self, password, pw_version):
        salt = secrets.token_hex(32)
        return dict(kdf="scrypt", salt=salt, hash=_scrypt(password, salt, **SCRYPT), pw_version=pw_version, changed=time.time(), **SCRYPT)

    def check_password(self, user, password):
        u = self.accounts["users"].get(user)
        if not u or not isinstance(password, str):
            _scrypt("x" * 12, "00" * 32, **SCRYPT)                    # same cost for unknown users
            return False
        return hmac.compare_digest(_scrypt(password, u["salt"], u["n"], u["r"], u["p"]), u["hash"])

    def change_password(self, user, current, new):
        with self.lock:
            if not self.check_password(user, current):
                return "CURRENT_PASSWORD_WRONG"
            p = password_problem(user, new)
            if p:
                return "WEAK_PASSWORD: " + p
            d = self.accounts
            d["users"][user] = self._hash_record(new, pw_version=int(d["users"][user]["pw_version"]) + 1)
            self._save_accounts(d)
            self.logout_all(user)
            return None

    def reset(self, user=USER):
        """Robot-side only (ctl auth-reset): remove the account and every session; re-enrollment needs a new setup code."""
        with self.lock:
            d = self.accounts
            d["users"].pop(user, None)
            self._save_accounts(d)
            self.logout_all(user)

    # ---------------- rate limits (memory: a restart does not unlock faster than the lock time matters)
    def locked(self, ip, user):
        now = time.time()
        fi, fu = self.fails_ip.get(ip, [0, 0]), self.fails_user.get(user, [0, 0])
        if fi[0] >= IP_FAILS and now - fi[1] < IP_LOCK_S:
            return f"LOCKED_OUT {IP_LOCK_S}s (이 기기에서 {IP_FAILS}번 실패)"
        if fu[0] >= USER_FAILS and now - fu[1] < USER_LOCK_S:
            return f"LOCKED_OUT {USER_LOCK_S}s (계정 실패 {USER_FAILS}번)"
        return None

    def note_fail(self, ip, user):
        now = time.time()
        for d, k, lock in ((self.fails_ip, ip, IP_LOCK_S), (self.fails_user, user, USER_LOCK_S)):
            n, t = d.get(k, [0, 0])
            d[k] = [1 if now - t > lock else n + 1, now]

    def note_ok(self, ip, user):
        self.fails_ip.pop(ip, None)
        self.fails_user.pop(user, None)

    # ---------------- sessions
    def login(self, ip, user, password, remember, ua=""):
        with self.lock:
            lk = self.locked(ip, user)
            if lk:
                return None, lk
            if self.needs_enrollment(USER):
                return None, "ENROLLMENT_REQUIRED"
            if user != USER or not self.check_password(user, password):
                self.note_fail(ip, user)
                return None, "WRONG_ID_OR_PASSWORD"
            self.note_ok(ip, user)
            return self.create(user, bool(remember), ua, ip), None

    def create(self, user, remember, ua, ip, kind="password", device=None):
        sid = secrets.token_urlsafe(32)
        now = time.time()
        pv = (self.accounts["users"].get(user) or {}).get("pw_version", 0)
        d = self._sessions()
        d["sessions"] = {k: v for k, v in d["sessions"].items() if v["expires"] > now}
        d["sessions"][self._h(sid)] = dict(user=user, created=now, expires=now + (REMEMBER_S if remember else SESSION_S),
                                          remember=remember, pw_version=pv, ua=(ua or "")[:120], ip_first=ip, last_seen=now, kind=kind,
                                          device=device)
        self._save_sessions(d)
        return dict(sid=sid, csrf=self.csrf(sid), max_age=REMEMBER_S if remember else None, user=user, remember=remember)

    def get(self, sid):
        """Valid session record or None. Checked on every request: expiry, account still present, password version."""
        if not sid:
            return None
        h = self._h(sid)
        s = self._sessions()["sessions"].get(h)
        if not s or s["expires"] <= time.time():
            return None
        if s.get("kind") in ("legacy_token", "device"):      # device sessions: bound to a Tailscale node, no password
            return dict(s, id=h)
        u = self.accounts["users"].get(s["user"])
        if not u or u["pw_version"] != s["pw_version"]:
            return None
        now = time.time()
        if now - self._seen_write.get(h, 0) > 300:                     # last_seen at most every 5 min (no write storm)
            with self.lock:
                d = self._sessions()
                if h in d["sessions"]:
                    d["sessions"][h]["last_seen"] = now
                    self._save_sessions(d)
            self._seen_write[h] = now
        return dict(s, id=h)

    def csrf(self, sid):
        return hmac.new(self.secret, b"csrf:" + sid.encode(), hashlib.sha256).hexdigest()

    def csrf_ok(self, sid, token):
        return bool(sid) and isinstance(token, str) and hmac.compare_digest(self.csrf(sid).encode(), token.encode())

    def logout(self, sid):
        with self.lock:
            d = self._sessions()
            d["sessions"].pop(self._h(sid or ""), None)
            self._save_sessions(d)

    def logout_all(self, user=USER):
        with self.lock:
            d = self._sessions()
            d["sessions"] = {k: v for k, v in d["sessions"].items() if v["user"] != user and v.get("kind") not in ("legacy_token", "device")}
            self._save_sessions(d)

    def devices(self, user, current_h=None):
        now = time.time()
        out = []
        for h, s in self._sessions()["sessions"].items():
            if s["user"] == user and s["expires"] > now:
                out.append(dict(id=h[:12], current=h == current_h, created=s["created"], last_seen=s.get("last_seen"),
                                expires=s["expires"], remember=s.get("remember"), ua=s.get("ua"), kind=s.get("kind")))
        return sorted(out, key=lambda x: -(x["last_seen"] or 0))

    def revoke(self, user, short_id):
        with self.lock:
            d = self._sessions()
            hit = [h for h, s in d["sessions"].items() if h.startswith(short_id) and s["user"] == user]
            for h in hit:
                d["sessions"].pop(h)
            self._save_sessions(d)
            return len(hit)
