# Copyright 2026 Kevin (AluminatiAI)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Stable machine identity for AluminatAI GPU Agent.

Generates a UUID the first time it runs and persists it to
~/.config/aluminatai/machine_id so it survives hostname changes and
process restarts.  On I/O failure (read-only FS, permission denied, etc.)
an ephemeral UUID is returned — the agent never crashes due to identity
issues.
"""
from __future__ import annotations

import uuid
from pathlib import Path

_MACHINE_ID_PATH = Path("~/.config/aluminatai/machine_id")


def get_machine_id() -> str:
    """Return a stable UUID string for this machine.

    Reads from ~/.config/aluminatai/machine_id if it exists; generates and
    persists a new UUID on first call.  Returns an ephemeral (non-persisted)
    UUID if any I/O error occurs.
    """
    path = _MACHINE_ID_PATH.expanduser()
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if path.exists():
            mid = path.read_text().strip()
            if mid:
                return mid
        mid = str(uuid.uuid4())
        path.write_text(mid)
        return mid
    except OSError:
        # Ephemeral fallback — non-fatal, agent continues running
        return str(uuid.uuid4())
