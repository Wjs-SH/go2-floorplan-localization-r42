"""When may E5 return the pan/tilt to HOME after an observation? (pure; replaces the student 'always /home' thread)

R1 (deploy src): HOME only if device_used AND device_idle_confirmed AND can_start_physical() (= web arm). When the web
trial was no longer walking, arm was False and the head stayed where it was (student fix ⑥).
Student 10-17: a daemon thread calls HTTP /home on the console whenever the device was used - bypassing the single
device owner (can race the next event's aim) and also when the device state is UNKNOWN (timeout, no idle proof).
E5RT2 REVIEW PATCH: cleanup is still motion. Unknown safety inputs, missing arm, and a mere boolean
cleanup_without_arm flag never authorize HOME. A future supervised event-scoped cleanup lease must be
implemented and verified separately before use. No such lease is created by this patch.
"""
from __future__ import annotations

HOME_OK = "HOME_CLEANUP_ALLOWED"


def home_decision(device_used, device_idle_confirmed, home_after=True, estop=None, operator_abort=None,
                  owner_busy_other=False, armed=False, cleanup_without_arm=False):
    if not device_used:
        return False, "NO_DEVICE_ACTION_TO_UNDO"
    if not home_after:
        return False, "HOME_AFTER_DISABLED"
    if estop is True or operator_abort is True:
        return False, "ESTOP_OR_OPERATOR_ABORT_NO_AUTO_MOTION"
    if estop is not False or operator_abort is not False:
        return False, "OPERATOR_SAFETY_STATE_UNKNOWN"
    if not device_idle_confirmed:
        return False, "DEVICE_STATE_UNKNOWN_NO_AUTO_HOME"
    if owner_busy_other:
        return False, "DEVICE_OWNED_BY_OTHER_JOB"
    if armed:
        return True, HOME_OK + ":armed"
    if cleanup_without_arm:
        # A boolean environment override cannot prove a current event-scoped cleanup lease.
        return False, "EVENT_SCOPED_CLEANUP_GRANT_REQUIRED"
    return False, "NOT_ARMED"
