"""test_other_account_gate.py — console.DeviceGate.identify 에 끼운 "추가 Tailscale 계정 허용" 덩어리 (학생 병합, 2026-11-15).

운영폴더 runtime/extra_tailscale_accounts 에 적힌 계정의 기기는 로봇 소유 계정의 기기와 똑같이 다룬다(보기 허용, 조작은 허용 목록의 기기만).
파일이 없으면 원래 규칙과 같아야 한다. tailscale 명령·인증 파일·네트워크는 쓰지 않는다(whois 와 허용 목록을 가짜로 넣는다)."""
import inspect
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "candidate/live1"))
sys.path.insert(0, str(HERE.parent / "candidate/live2"))

ROBOT, OWNER, FRIEND, STRANGER = "<ROBOT_TAILSCALE_IP>", "owner@example.com", "friend@example.com", "stranger@example.com"
PHONE_IP, SID = "100.70.1.2", "nPHONE0000CNTRL"


def gate(tmp_path, monkeypatch, whois, trusted, accounts=None):
    import console as CO
    (tmp_path / "runtime").mkdir(exist_ok=True)
    if accounts is not None:
        (tmp_path / "runtime" / "extra_tailscale_accounts").write_text(accounts)
    monkeypatch.setattr(CO, "L", tmp_path)
    g = CO.DeviceGate.__new__(CO.DeviceGate)
    g.self_ips, g.owner = {ROBOT}, OWNER
    g.whois = lambda ip: whois
    g.trusted = lambda: trusted
    return g


def who(user, sid=SID, ip=PHONE_IP):
    return dict(stable_id=sid, name="phone", fqdn="phone.example.ts.net", user=user, tags=[], addrs=[ip])


def patched():
    import console as CO
    return "extra_tailscale_accounts" in inspect.getsource(CO.DeviceGate.identify)


needs_block = pytest.mark.skipif("not patched()", reason="이 console.py 에는 '추가 계정 허용' 덩어리가 없다")
TRUST = {SID: dict(stable_id=SID, name="phone")}
ONE = "# 추가로 받는 Tailscale 계정\nfriend@example.com\n"


# ---- 원래 규칙 (덩어리가 있든 없든, 계정 파일이 있든 없든 같아야 한다)
@pytest.mark.parametrize("accounts", [None, "", ONE])
def test_owner_account_is_unchanged(tmp_path, monkeypatch, accounts):
    r = gate(tmp_path, monkeypatch, who(OWNER), TRUST, accounts).identify(PHONE_IP, ROBOT)
    assert r["read"] is True and r["control"] is True and r["why"] is None and r["actor"] == "dev:" + SID
    r = gate(tmp_path, monkeypatch, who(OWNER), {}, accounts).identify(PHONE_IP, ROBOT)
    assert r["read"] is True and r["control"] is False and "DEVICE_NOT_TRUSTED" in r["why"]


@pytest.mark.parametrize("accounts", [None, "", "# friend@example.com\n", "   \n\n"])
def test_other_account_gets_nothing_without_a_listed_account(tmp_path, monkeypatch, accounts):
    for trusted in ({}, TRUST):                       # 허용 목록에 기기가 올라 있어도 계정이 안 맞으면 아무것도 못 한다
        r = gate(tmp_path, monkeypatch, who(FRIEND), trusted, accounts).identify(PHONE_IP, ROBOT)
        assert r["read"] is False and r["control"] is False and r["why"] == "OTHER_TAILNET_USER"


@pytest.mark.parametrize("accounts", [None, ONE])
def test_lan_wrong_local_address_and_whois_failure_get_nothing(tmp_path, monkeypatch, accounts):
    g = gate(tmp_path, monkeypatch, who(FRIEND), TRUST, accounts)
    assert g.identify("192.168.0.4", ROBOT)["read"] is False                          # 같은 와이파이 주소
    assert g.identify(PHONE_IP, "192.168.0.10")["why"] == "NOT_RECEIVED_ON_TAILSCALE_ADDRESS"
    assert gate(tmp_path, monkeypatch, None, TRUST, accounts).identify(PHONE_IP, ROBOT)["why"] == "TAILSCALE_WHOIS_FAILED"
    g3 = gate(tmp_path, monkeypatch, who(FRIEND, ip="100.70.9.9"), TRUST, accounts)     # whois 가 말한 주소와 접속 주소가 다름
    assert g3.identify(PHONE_IP, ROBOT)["why"] == "TAILSCALE_WHOIS_FAILED"


# ---- 끼운 덩어리
@needs_block
def test_listed_account_untrusted_device_is_read_only(tmp_path, monkeypatch):
    r = gate(tmp_path, monkeypatch, who(FRIEND), {}, ONE).identify(PHONE_IP, ROBOT)
    assert r["read"] is True and r["control"] is False and "DEVICE_NOT_TRUSTED" in r["why"] and r["user"] == FRIEND


@needs_block
def test_listed_account_trusted_device_can_control(tmp_path, monkeypatch):
    r = gate(tmp_path, monkeypatch, who(FRIEND), TRUST, ONE).identify(PHONE_IP, ROBOT)
    assert r["read"] is True and r["control"] is True and r["why"] is None and r["actor"] == "dev:" + SID


@needs_block
def test_account_not_in_the_file_still_gets_nothing(tmp_path, monkeypatch):
    r = gate(tmp_path, monkeypatch, who(STRANGER), TRUST, ONE).identify(PHONE_IP, ROBOT)
    assert r["read"] is False and r["control"] is False and r["why"] == "OTHER_TAILNET_USER"


@needs_block
@pytest.mark.parametrize("user", ["", None])
def test_peer_without_a_login_name_gets_nothing(tmp_path, monkeypatch, user):
    r = gate(tmp_path, monkeypatch, who(user), TRUST, ONE + "\n\n").identify(PHONE_IP, ROBOT)
    assert r["read"] is False and r["control"] is False


@needs_block
def test_account_match_ignores_case_and_spaces_but_not_substrings(tmp_path, monkeypatch):
    acc = "  Friend@Example.com  \n"
    assert gate(tmp_path, monkeypatch, who(FRIEND), TRUST, acc).identify(PHONE_IP, ROBOT)["control"] is True
    assert gate(tmp_path, monkeypatch, who("xfriend@example.com"), TRUST, acc).identify(PHONE_IP, ROBOT)["read"] is False
    assert gate(tmp_path, monkeypatch, who("friend@example.co"), TRUST, acc).identify(PHONE_IP, ROBOT)["read"] is False


@needs_block
def test_removing_the_file_restores_the_original_rule(tmp_path, monkeypatch):
    g = gate(tmp_path, monkeypatch, who(FRIEND), TRUST, ONE)
    assert g.identify(PHONE_IP, ROBOT)["control"] is True
    (tmp_path / "runtime" / "extra_tailscale_accounts").unlink()
    assert g.identify(PHONE_IP, ROBOT)["why"] == "OTHER_TAILNET_USER"
