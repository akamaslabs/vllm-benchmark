"""Run the AWS CLI against the lab account and turn its output into JSON or AwsError."""
import json
import os
import re
import subprocess
from typing import Any, Callable

PROFILE = "lab"  # the default profile on this workstation points at another account
CLI_TIMEOUT_S = 120

Call = Callable[[list[str], "str | None"], Any]


class AwsError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def parse_error(stderr: str) -> AwsError:
    m = re.search(r"An error occurred \(([A-Za-z0-9.]+)\)[^:]*:\s*(.*)", stderr, re.S)
    if m:
        return AwsError(m.group(1), m.group(2).strip())
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    return AwsError("CliError", lines[-1] if lines else "aws CLI failed without a message")


def aws(args: list[str], region: str | None = None) -> Any:
    # --profile on the command line, not AWS_PROFILE: exported access keys would win over the
    # env var and send reservations (which cost money) to another account.
    argv = ["aws", *args, "--output", "json", "--profile", PROFILE]
    if region:
        argv += ["--region", region]
    env = {**os.environ, "AWS_PAGER": ""}
    try:
        # Own session: a Ctrl+C in the terminal must not kill a create half-way (the id would be
        # lost); no stdin: a credential prompt fails instead of hanging a probe thread.
        proc = subprocess.run(argv, capture_output=True, text=True, env=env, stdin=subprocess.DEVNULL,
                              start_new_session=True, timeout=CLI_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise AwsError("Timeout", f"aws {' '.join(args[:2])} did not answer in {CLI_TIMEOUT_S} s")
    if proc.returncode != 0:
        raise parse_error(proc.stderr)
    out = proc.stdout.strip()
    return json.loads(out) if out else {}
