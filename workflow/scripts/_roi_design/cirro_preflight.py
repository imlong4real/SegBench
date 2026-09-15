#!/usr/bin/env python3
"""Fail fast when Cirro credentials are missing, instead of hanging.

The SDK's device-code flow blocks indefinitely waiting for a browser login it
prints to stdout. In an unattended run that looks identical to a slow API: the
call simply never returns, and a campaign can sit silently for a day before
anyone notices. This checks for usable credentials first and exits with the
exact remediation command.

Import `require_credentials()` at the top of any script that talks to Cirro, or
run this module directly as a standalone check.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

VENV = "/scratch4/adeshpa6/segbench_envs/venvs/cirro-sdk-py311"
REMEDY = f"""Cirro has no usable credentials, and the SDK would block on an
interactive device-code login rather than returning an error.

Fix it one of two ways:

  1. Interactive device login (per-machine, expires):
       {VENV}/bin/cirro-cli login
     or, if that subcommand is unavailable:
       {VENV}/bin/python -c "from cirro import DataPortal; DataPortal()"
     then open the printed URL in a browser.

  2. Non-expiring client credentials (better for multi-day campaigns):
       export CIRRO_CLIENT_ID=...
       export CIRRO_CLIENT_SECRET=...
"""


def credentials_present() -> tuple[bool, str]:
    """Return (ok, description of what was found)."""
    if os.getenv("CIRRO_CLIENT_ID") and os.getenv("CIRRO_CLIENT_SECRET"):
        return True, "client credentials from CIRRO_CLIENT_ID/CIRRO_CLIENT_SECRET"
    home = Path(os.environ.get("CIRRO_HOME", "~/.cirro")).expanduser()
    tokens = sorted(home.glob("*.token.dat"))
    if tokens:
        return True, f"cached device token {tokens[0].name}"
    return False, f"no *.token.dat in {home} and no client-credential env vars"


def require_credentials() -> None:
    ok, what = credentials_present()
    if not ok:
        sys.stderr.write(f"\n{REMEDY}\nChecked: {what}\n")
        raise SystemExit(3)


if __name__ == "__main__":
    ok, what = credentials_present()
    print(("OK: " if ok else "MISSING: ") + what)
    if not ok:
        sys.stderr.write(f"\n{REMEDY}")
    sys.exit(0 if ok else 3)
