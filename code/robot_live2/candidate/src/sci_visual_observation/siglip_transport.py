"""R1 deadline-bounded framed pipe transport for the SigLIP worker (length-prefixed pickle frames).

The legacy client stored `timeout` but performed blocking reads/writes with no deadline. Here every write and read
has a real deadline (select on non-blocking fds). After a timeout or an invalid frame the transport is POISONED:
the byte stream position is unknown, so a late reply can never be read as the answer to the next request. The
caller must discard the worker (the E5 node restarts it and logs the scene as skipped).
"""
from __future__ import annotations

import os
import pickle
import select
import struct
import time


class TransportError(RuntimeError):
    pass


class TransportTimeout(TransportError):
    pass


class PipeTransport:
    def __init__(self, rfd: int, wfd: int, max_frame=64 << 20, clock=time.monotonic):
        self.rfd, self.wfd, self.max_frame, self.clock = rfd, wfd, max_frame, clock
        os.set_blocking(rfd, False)
        os.set_blocking(wfd, False)
        self.poisoned = None

    def _check(self):
        if self.poisoned:
            raise TransportError(f"transport poisoned ({self.poisoned}); restart the worker")

    def _poison(self, why, exc):
        self.poisoned = why
        raise exc(why)

    def send(self, obj, deadline_s):
        self._check()
        b = pickle.dumps(obj, protocol=4)
        data = memoryview(struct.pack("<I", len(b)) + b)
        t_end = self.clock() + deadline_s
        while data:
            left = t_end - self.clock()
            if left <= 0:
                self._poison("write_deadline", TransportTimeout)
            _, w, _ = select.select([], [self.wfd], [], left)
            if not w:
                continue
            try:
                n = os.write(self.wfd, data)
            except BlockingIOError:
                continue
            except (BrokenPipeError, OSError) as e:
                self._poison(f"write_error:{e}", TransportError)
            data = data[n:]

    def _read_exact(self, n, t_end):
        buf = bytearray()
        while len(buf) < n:
            left = t_end - self.clock()
            if left <= 0:
                self._poison("read_deadline", TransportTimeout)
            r, _, _ = select.select([self.rfd], [], [], left)
            if not r:
                continue
            try:
                chunk = os.read(self.rfd, n - len(buf))
            except BlockingIOError:
                continue
            if not chunk:
                self._poison("worker_closed_pipe", TransportError)
            buf += chunk
        return bytes(buf)

    def recv(self, deadline_s):
        self._check()
        t_end = self.clock() + deadline_s
        (n,) = struct.unpack("<I", self._read_exact(4, t_end))
        if n > self.max_frame:
            self._poison(f"frame_too_large:{n}", TransportError)
        raw = self._read_exact(n, t_end)
        try:
            obj = pickle.loads(raw)
        except Exception as e:
            self._poison(f"invalid_reply:{type(e).__name__}", TransportError)
        if not isinstance(obj, dict):
            self._poison("invalid_reply:not_a_dict", TransportError)
        return obj

    def request(self, obj, deadline_s):
        """send + recv under ONE overall deadline."""
        t0 = self.clock()
        self.send(obj, deadline_s)
        return self.recv(max(0.0, deadline_s - (self.clock() - t0)))
