#!/usr/bin/env python3
"""Read-only environment probe for the E5-E7 research code (imports only, no models, no ROS graph changes)."""
import importlib
import sys

print("python", sys.version.split()[0], sys.executable)
for name in ("numpy", "scipy", "cv2", "yaml", "rclpy", "PIL", "shapely", "sklearn"):
    try:
        m = importlib.import_module(name)
        print(f"{name:8s} OK  {getattr(m, '__version__', '')}")
    except Exception as e:  # noqa: BLE001
        print(f"{name:8s} MISSING  {type(e).__name__}: {e}")
