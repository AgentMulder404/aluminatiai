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
GPU Attribution Engine

Resolves which job(s) own a GPU at each sample and splits power
proportionally by GPU memory fraction when multiple processes share a device.
"""

from .engine import AttributionEngine, AttributionResult
from .rules import AttributionRules, AttributionRule

__all__ = ["AttributionEngine", "AttributionResult", "AttributionRules", "AttributionRule"]
