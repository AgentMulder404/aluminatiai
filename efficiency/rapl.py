"""
RAPL energy reader — CPU + DRAM power monitoring via sysfs.

Reads /sys/class/powercap/{intel-rapl,amd_rapl}:*/energy_uj counters with
overflow handling. Supports multi-socket systems (one RaplPackage per socket).
Auto-disabled on non-Linux or when sysfs is not readable.

Usage:
    reader = RaplReader()
    if reader.available:
        snapshot = reader.read()
        # ... wait ...
        delta = reader.read()
        for pkg in delta:
            print(f"Socket {pkg.package_index}: {pkg.package_watts:.1f}W CPU, {pkg.dram_watts:.1f}W RAM")
"""
from __future__ import annotations

import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

_RAPL_BASE = Path("/sys/class/powercap")

# Legacy single-package dataclass (kept for backward compat)
@dataclass
class RaplReading:
    package_energy_uj: int
    dram_energy_uj: int
    timestamp: float
    package_watts: float = 0.0
    dram_watts: float = 0.0


@dataclass
class RaplPackageReading:
    """Per-socket RAPL reading with optional core/uncore/dram subdomains."""
    package_index: int
    package_energy_uj: int
    timestamp: float
    package_watts: float = 0.0
    dram_energy_uj: int = 0
    dram_watts: float = 0.0
    core_energy_uj: int = 0
    core_watts: float = 0.0
    uncore_energy_uj: int = 0
    uncore_watts: float = 0.0
    power_limit_w: float = 0.0
    cpu_model: str = ""


@dataclass
class _PackageState:
    """Internal state for one RAPL package."""
    index: int
    prefix: str                          # "intel-rapl" or "amd_rapl"
    package_path: Path
    dram_path: Optional[Path] = None
    core_path: Optional[Path] = None
    uncore_path: Optional[Path] = None
    max_energy_uj: int = 2**32
    power_limit_w: float = 0.0
    last: Optional[RaplPackageReading] = field(default=None, repr=False)


class RaplReader:
    """Read RAPL energy counters from sysfs (Intel + AMD, multi-socket)."""

    def __init__(self):
        self._packages: List[_PackageState] = []
        self._available = False
        self._cpu_model = ""

        # Legacy single-package state for backward compat
        self._package_path: Optional[Path] = None
        self._dram_path: Optional[Path] = None
        self._max_energy_uj: int = 0
        self._last: Optional[RaplReading] = None

        if not sys.platform.startswith("linux"):
            logger.debug("RAPL unavailable: not Linux")
            return

        self._cpu_model = self._detect_cpu_model()
        self._discover_packages()

        if self._packages:
            self._available = True
            # Set legacy fields from first package for backward compat
            self._package_path = self._packages[0].package_path
            self._dram_path = self._packages[0].dram_path
            self._max_energy_uj = self._packages[0].max_energy_uj

            names = [f"pkg{p.index}({p.prefix})" for p in self._packages]
            logger.info("RAPL enabled: %d package(s) [%s], cpu=%s",
                        len(self._packages), ", ".join(names),
                        self._cpu_model or "unknown")

    def _detect_cpu_model(self) -> str:
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except (OSError, PermissionError):
            pass
        return ""

    def _discover_packages(self) -> None:
        """Scan sysfs for all RAPL packages (intel-rapl:N and amd_rapl:N)."""
        if not _RAPL_BASE.exists():
            logger.debug("RAPL unavailable: %s not found", _RAPL_BASE)
            return

        # Match intel-rapl:N or amd_rapl:N (top-level packages only, no subdomain colons)
        pkg_pattern = re.compile(r"^(intel-rapl|amd_rapl):(\d+)$")

        for entry in sorted(_RAPL_BASE.iterdir()):
            m = pkg_pattern.match(entry.name)
            if not m:
                continue

            prefix = m.group(1)
            pkg_index = int(m.group(2))
            energy_path = entry / "energy_uj"

            if not energy_path.exists():
                continue

            try:
                energy_path.read_text()
            except PermissionError:
                logger.debug("RAPL: permission denied on %s", energy_path)
                continue

            state = _PackageState(
                index=pkg_index,
                prefix=prefix,
                package_path=energy_path,
            )

            # Discover subdomains (dram, core, uncore)
            for subdir in sorted(entry.iterdir()):
                name_file = subdir / "name"
                if not name_file.exists():
                    continue
                try:
                    name = name_file.read_text().strip()
                except (PermissionError, OSError):
                    continue

                sub_energy = subdir / "energy_uj"
                if not sub_energy.exists():
                    continue

                if name == "dram":
                    state.dram_path = sub_energy
                elif name == "core":
                    state.core_path = sub_energy
                elif name == "uncore":
                    state.uncore_path = sub_energy

            # Max energy range for overflow
            max_path = entry / "max_energy_range_uj"
            if max_path.exists():
                try:
                    state.max_energy_uj = int(max_path.read_text().strip())
                except (ValueError, PermissionError):
                    pass

            # Power limit (constraint_0 is the sustained/long-term limit)
            limit_path = entry / "constraint_0_power_limit_uw"
            if limit_path.exists():
                try:
                    state.power_limit_w = int(limit_path.read_text().strip()) / 1_000_000
                except (ValueError, PermissionError):
                    pass

            self._packages.append(state)
            logger.debug("RAPL: found %s:%d dram=%s core=%s uncore=%s limit=%.0fW",
                         prefix, pkg_index,
                         "yes" if state.dram_path else "no",
                         "yes" if state.core_path else "no",
                         "yes" if state.uncore_path else "no",
                         state.power_limit_w)

    @property
    def available(self) -> bool:
        return self._available

    @property
    def package_count(self) -> int:
        return len(self._packages)

    @property
    def cpu_model(self) -> str:
        return self._cpu_model

    def read_all(self) -> List[RaplPackageReading]:
        """Read RAPL counters for all packages. Returns one reading per socket."""
        if not self._available:
            return []

        now = time.monotonic()
        results: List[RaplPackageReading] = []

        for pkg in self._packages:
            try:
                pkg_uj = int(pkg.package_path.read_text().strip())
            except (ValueError, PermissionError, OSError):
                continue

            dram_uj = self._read_counter(pkg.dram_path)
            core_uj = self._read_counter(pkg.core_path)
            uncore_uj = self._read_counter(pkg.uncore_path)

            reading = RaplPackageReading(
                package_index=pkg.index,
                package_energy_uj=pkg_uj,
                dram_energy_uj=dram_uj,
                core_energy_uj=core_uj,
                uncore_energy_uj=uncore_uj,
                timestamp=now,
                power_limit_w=pkg.power_limit_w,
                cpu_model=self._cpu_model,
            )

            if pkg.last is not None:
                dt = now - pkg.last.timestamp
                if dt > 0:
                    reading.package_watts = self._delta_with_overflow(
                        pkg.last.package_energy_uj, pkg_uj, pkg.max_energy_uj
                    ) / (dt * 1_000_000)
                    reading.dram_watts = self._delta_with_overflow(
                        pkg.last.dram_energy_uj, dram_uj, pkg.max_energy_uj
                    ) / (dt * 1_000_000)
                    reading.core_watts = self._delta_with_overflow(
                        pkg.last.core_energy_uj, core_uj, pkg.max_energy_uj
                    ) / (dt * 1_000_000)
                    reading.uncore_watts = self._delta_with_overflow(
                        pkg.last.uncore_energy_uj, uncore_uj, pkg.max_energy_uj
                    ) / (dt * 1_000_000)

            pkg.last = reading
            results.append(reading)

        return results

    def read(self) -> Optional[RaplReading]:
        """Read RAPL counters (legacy single-package interface).

        Returns aggregate watts across all packages for backward compat.
        """
        readings = self.read_all()
        if not readings:
            return None

        now = readings[0].timestamp
        total_pkg_uj = sum(r.package_energy_uj for r in readings)
        total_dram_uj = sum(r.dram_energy_uj for r in readings)
        total_pkg_w = sum(r.package_watts for r in readings)
        total_dram_w = sum(r.dram_watts for r in readings)

        return RaplReading(
            package_energy_uj=total_pkg_uj,
            dram_energy_uj=total_dram_uj,
            timestamp=now,
            package_watts=total_pkg_w,
            dram_watts=total_dram_w,
        )

    @staticmethod
    def _read_counter(path: Optional[Path]) -> int:
        if path is None:
            return 0
        try:
            return int(path.read_text().strip())
        except (ValueError, PermissionError, OSError):
            return 0

    @staticmethod
    def _delta_with_overflow(prev: int, curr: int, max_uj: int = 2**32) -> int:
        """Handle counter overflow (wraps at max_energy_range_uj)."""
        if curr >= prev:
            return curr - prev
        return (max_uj - prev) + curr
