"""Filesystem locations shared by process control and the application."""

import os
from pathlib import Path


AGENT_WORKDIR = Path(
    os.getenv(
        "AGENT_WORKDIR",
        ".",
    )
).expanduser().resolve()

RAPTOR_HOME = Path(
    os.getenv(
        "RAPTOR_HOME",
        str(AGENT_WORKDIR / ".raptor"),
    )
).expanduser().resolve()

CHAT_DIR = RAPTOR_HOME / "chats"
STATE_PATH = RAPTOR_HOME / "state.json"
LOG_PATH = Path(
    os.getenv(
        "RAPTOR_LOG",
        str(RAPTOR_HOME / "raptor.log"),
    )
).expanduser().resolve()
