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
Scheduler adapters for GPU job attribution.

Detects which scheduler is managing compute resources and intercepts
job metadata to link GPU power metrics to specific jobs, teams, and models.
"""

from .base import SchedulerAdapter, JobMetadata, NullAdapter
from .detect import detect_scheduler

__all__ = [
    'SchedulerAdapter',
    'JobMetadata',
    'NullAdapter',
    'detect_scheduler',
]
