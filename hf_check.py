"""Check whether the stored HF token can WRITE (needed to upload the checkpoint).

The laptop's TLS interception kills the 865 MB Lightning download
(DECRYPTION_FAILED_OR_BAD_RECORD_MAC), so the plan is: studio -> HF Hub -> laptop.
That requires a write-scoped token; this one was only ever used for reads.

Prints the role only, never the token.
"""
import pathlib
import re

import requests

MEM = pathlib.Path.home() / ".claude/projects/D--Hyperframe/memory/heavens-gate-lora-hf-token.md"


def token():
    m = re.search(r"hf_[A-Za-z0-9]+", MEM.read_text(encoding="utf-8"))
    if not m:
        raise SystemExit("no token found in memory file")
    return m.group(0)


def main():
    t = token()
    r = requests.get(
        "https://huggingface.co/api/whoami-v2",
        headers={"Authorization": f"Bearer {t}"},
        timeout=30,
    )
    r.raise_for_status()
    d = r.json()
    auth = d.get("auth", {})
    at = auth.get("accessToken", {})
    print("user      :", d.get("name"))
    print("token role:", at.get("role"))
    print("token name:", at.get("displayName"))
    can_write = at.get("role") == "write"
    print("CAN WRITE :", can_write)
    if not can_write:
        print("\nNeed a write-scoped token to upload the checkpoint.")
        print("Create one at https://huggingface.co/settings/tokens (role: Write)")


if __name__ == "__main__":
    main()
