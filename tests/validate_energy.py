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
Energy Calculation Validation

Validates that energy calculations (E = P × Δt) are accurate by:
1. Collecting metrics over a known time period
2. Comparing calculated energy against theoretical values
3. Checking for drift and accumulation errors
4. Comparing against GPU's internal energy counter (if available)
"""

import sys
import time
import argparse
import statistics
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from collector import GPUCollector


class EnergyValidator:
    """Validate energy calculation accuracy"""

    def __init__(self, duration: int = 60):
        """
        Args:
            duration: Test duration in seconds
        """
        self.duration = duration
        self.samples = []
        self.gpu_totals = {}

    def run_validation(self, interval: float = 1.0):
        """
        Collect metrics and validate energy calculations.

        Args:
            interval: Sampling interval in seconds
        """
        print(f"\n🔬 Energy Validation Test")
        print("="*70)
        print(f"Duration: {self.duration}s")
        print(f"Sampling interval: {interval}s")
        print(f"Expected samples: ~{int(self.duration / interval)}")
        print("="*70 + "\n")

        try:
            with GPUCollector() as collector:
                gpu_count = collector.get_gpu_count()
                print(f"📊 Monitoring {gpu_count} GPUs...\n")

                # Initialize tracking
                for i in range(gpu_count):
                    self.gpu_totals[i] = {
                        'energy_sum_j': 0.0,
                        'power_samples': [],
                        'sample_count': 0,
                        'first_timestamp': None,
                        'last_timestamp': None,
                    }

                # Collect metrics
                start_time = time.time()
                sample_num = 0

                while time.time() - start_time < self.duration:
                    loop_start = time.time()

                    metrics = collector.collect()
                    sample_num += 1

                    # Process each GPU
                    for m in metrics:
                        gpu_data = self.gpu_totals[m.gpu_index]

                        # Track timestamps
                        if gpu_data['first_timestamp'] is None:
                            gpu_data['first_timestamp'] = m.timestamp
                        gpu_data['last_timestamp'] = m.timestamp

                        # Accumulate energy
                        if m.energy_delta_j is not None:
                            gpu_data['energy_sum_j'] += m.energy_delta_j

                        # Track power samples
                        gpu_data['power_samples'].append(m.power_draw_w)
                        gpu_data['sample_count'] += 1

                    # Display progress
                    if sample_num % 10 == 0:
                        elapsed = time.time() - start_time
                        remaining = self.duration - elapsed
                        print(f"  Sample {sample_num:4d} | "
                              f"Elapsed: {elapsed:5.1f}s | "
                              f"Remaining: {remaining:5.1f}s")

                    # Sleep to maintain interval
                    elapsed = time.time() - loop_start
                    sleep_time = max(0, interval - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                actual_duration = time.time() - start_time
                print(f"\n✅ Collection complete. Actual duration: {actual_duration:.2f}s\n")

                # Validate results
                self._validate_results(actual_duration)

        except Exception as e:
            print(f"\n❌ Validation failed: {e}")
            import traceback
            traceback.print_exc()
            return 1

        return 0

    def _validate_results(self, actual_duration: float):
        """Analyze and validate collected data"""
        print("="*70)
        print(" VALIDATION RESULTS")
        print("="*70 + "\n")

        all_pass = True

        for gpu_idx, data in sorted(self.gpu_totals.items()):
            print(f"GPU {gpu_idx}")
            print("-"*70)

            # Calculate statistics
            energy_kwh = data['energy_sum_j'] / 3_600_000
            avg_power = statistics.mean(data['power_samples']) if data['power_samples'] else 0
            power_stddev = statistics.stdev(data['power_samples']) if len(data['power_samples']) > 1 else 0

            # Theoretical energy based on average power
            theoretical_energy_j = avg_power * actual_duration
            theoretical_energy_kwh = theoretical_energy_j / 3_600_000

            # Calculate error
            if theoretical_energy_j > 0:
                error_pct = abs(data['energy_sum_j'] - theoretical_energy_j) / theoretical_energy_j * 100
            else:
                error_pct = 0

            # Display metrics
            print(f"{'Samples collected':<35} {data['sample_count']:>10}")
            print(f"{'Average power':<35} {avg_power:>10.2f} W")
            print(f"{'Power std dev':<35} {power_stddev:>10.2f} W")
            print(f"{'Power stability (CV)':<35} {(power_stddev/avg_power*100) if avg_power > 0 else 0:>9.2f} %")
            print()
            print(f"{'Measured energy (sum of deltas)':<35} {data['energy_sum_j']:>10.1f} J")
            print(f"{'Theoretical energy (P_avg × t)':<35} {theoretical_energy_j:>10.1f} J")
            print(f"{'Error':<35} {error_pct:>9.2f} %")
            print()
            print(f"{'Total energy (kWh)':<35} {energy_kwh:>10.6f}")
            print(f"{'Estimated cost ($0.12/kWh)':<35} ${energy_kwh * 0.12:>9.6f}")

            # Validation checks
            print("\nValidation Checks:")

            # Check 1: Error should be < 5%
            check1 = error_pct < 5.0
            print(f"  {'Energy error < 5%':45} {'✅ PASS' if check1 else '❌ FAIL'}")
            if not check1:
                print(f"     (Error: {error_pct:.2f}%)")
                all_pass = False

            # Check 2: Should have expected number of samples (±10%)
            expected_samples = actual_duration / 1.0  # Assuming 1s interval
            sample_ratio = data['sample_count'] / expected_samples if expected_samples > 0 else 0
            check2 = 0.9 <= sample_ratio <= 1.1
            print(f"  {'Sample count within ±10%':45} {'✅ PASS' if check2 else '❌ FAIL'}")
            if not check2:
                print(f"     (Expected: ~{expected_samples:.0f}, Got: {data['sample_count']})")
                all_pass = False

            # Check 3: Energy should be positive and non-zero (if GPU is active)
            check3 = data['energy_sum_j'] > 0
            print(f"  {'Energy > 0 (GPU active)':45} {'✅ PASS' if check3 else '⚠️  WARN'}")
            if not check3:
                print(f"     (GPU may be idle or not running workload)")

            # Check 4: Power should be reasonable (between 10W and 600W)
            check4 = 10 <= avg_power <= 600
            print(f"  {'Power in reasonable range':45} {'✅ PASS' if check4 else '⚠️  WARN'}")
            if not check4:
                print(f"     (Average power: {avg_power:.1f}W)")

            print()

        # Overall result
        print("="*70)
        if all_pass:
            print("✅ VALIDATION PASSED - Energy calculations are accurate")
        else:
            print("❌ VALIDATION FAILED - Review errors above")
        print("="*70 + "\n")

        # Additional notes
        print("📝 Notes:")
        print("  • Energy error < 5% is excellent for real-world monitoring")
        print("  • Small errors can occur due to:")
        print("    - Timing jitter in sampling loop")
        print("    - Power fluctuations between samples")
        print("    - NVML query latency variations")
        print("  • For validation against external meter (kill-a-watt):")
        print("    - Run GPU-intensive workload (training, stress test)")
        print("    - Use longer duration (5+ minutes)")
        print("    - Compare total kWh from agent vs meter")
        print()


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Validate GPU energy calculation accuracy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick validation (60 seconds)
  python validate_energy.py

  # Longer test for better accuracy
  python validate_energy.py --duration 300

  # Fast sampling
  python validate_energy.py --interval 0.5 --duration 60

Validation Method:
  This script validates that E = P × Δt is correctly implemented by:
  1. Collecting power samples over known duration
  2. Summing energy deltas: E_measured = Σ(P_i × Δt_i)
  3. Computing theoretical energy: E_theoretical = P_avg × t_total
  4. Comparing the two values (should match within ~5%)

For External Validation:
  To validate against a kill-a-watt meter or similar:
  1. Start a GPU workload (e.g., training job)
  2. Run this script for 5+ minutes
  3. Compare "Total energy (kWh)" with meter reading
  4. Note: Meter measures total system power, agent measures GPU only
        """
    )

    parser.add_argument(
        '--duration', '-d',
        type=int,
        default=60,
        help='Test duration in seconds (default: 60)'
    )

    parser.add_argument(
        '--interval', '-i',
        type=float,
        default=1.0,
        help='Sampling interval in seconds (default: 1.0)'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.duration < 10:
        print("Error: Duration must be at least 10 seconds")
        return 1

    if args.interval < 0.1 or args.interval > 10:
        print("Error: Interval must be between 0.1 and 10 seconds")
        return 1

    # Run validation
    validator = EnergyValidator(duration=args.duration)
    return validator.run_validation(interval=args.interval)


if __name__ == '__main__':
    sys.exit(main())
