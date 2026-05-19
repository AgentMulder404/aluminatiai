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
#
# AluminatiAI — https://github.com/AgentMulder404/AluminatAI
"""
Custom attribution rules loader.

Operators can define their own cmdline→team mappings in a JSON config file
so that workloads not covered by built-in heuristics are correctly attributed.

Config file format:
  {
    "rules": [
      { "pattern": "python.*gpt4_train", "team": "llm-infra", "model": "gpt4",    "priority": 10 },
      { "pattern": "vllm.*llama",        "team": "inference",  "model": "llama",   "priority": 5  },
      { "pattern": "jupyter",            "team": "research",   "model": "notebook", "priority": 1 }
    ]
  }

Config file search order:
  1. ALUMINATAI_ATTRIBUTION_CONFIG env var (explicit path)
  2. ./attribution_rules.json  (working directory)
  3. ~/.config/aluminatai/attribution_rules.json
  4. Not found → rules disabled silently (no behaviour change)
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AttributionRule:
    pattern: str
    team: str
    model: str = "untagged"
    priority: int = 0   # higher priority rules are checked first


class AttributionRules:
    """
    Loads and matches custom attribution rules from a JSON config file.

    Usage:
        rules = AttributionRules()
        rules.load()                       # idempotent, silent if no file
        match = rules.match(cmdline)       # returns AttributionRule or None
    """

    def __init__(self) -> None:
        self._rules: list[tuple[re.Pattern, AttributionRule]] = []

    def load(self) -> None:
        """
        Load rules from the config file. Silent no-op if no file is found.
        Logs a warning if the file exists but is malformed.
        """
        path = self._find_config_file()
        if path is None:
            return

        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("attribution rules: could not load %s: %s", path, exc)
            return

        compiled: list[tuple[re.Pattern, AttributionRule]] = []
        for item in data.get("rules", []):
            try:
                rule = AttributionRule(
                    pattern=item["pattern"],
                    team=item["team"],
                    model=item.get("model", "untagged"),
                    priority=int(item.get("priority", 0)),
                )
                compiled.append((re.compile(rule.pattern), rule))
            except (KeyError, re.error, ValueError) as exc:
                logger.warning("attribution rules: skipping invalid rule %r: %s", item, exc)

        # Higher priority checked first
        compiled.sort(key=lambda x: x[1].priority, reverse=True)
        self._rules = compiled
        logger.info("attribution rules: loaded %d rule(s) from %s", len(compiled), path)

    def match(self, cmdline: str) -> Optional[AttributionRule]:
        """Return the highest-priority rule that matches `cmdline`, or None."""
        if not cmdline:
            return None
        for compiled_pattern, rule in self._rules:
            try:
                if compiled_pattern.search(cmdline):
                    return rule
            except (TypeError, re.error):
                continue
        return None

    def _find_config_file(self) -> Optional[str]:
        """Search standard locations for the rules config file."""
        # 1. Explicit env var (read dynamically so tests can patch os.environ)
        explicit = os.getenv("ALUMINATAI_ATTRIBUTION_CONFIG", "")
        if explicit and os.path.isfile(explicit):
            return explicit

        # 2. Working directory
        local = "./attribution_rules.json"
        if os.path.isfile(local):
            return local

        # 3. User config directory
        user = os.path.expanduser("~/.config/aluminatai/attribution_rules.json")
        if os.path.isfile(user):
            return user

        return None
