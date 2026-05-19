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
Benchmark CPU and Memory Overhead of GPU Energy Agent

Measures:
1. Baseline resource usage (no agent running)
2. Agent resource usage at various sampling intervals
3. Per-GPU overhead
4. Collection latency (time per sample)
"""

import os
import sys
import time
import psutil
import statistics
from multiprocessing import Process
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from collector import GPUCollector


class OverheadBenchmark:
    """Measure agent overhead"""

    def __init__(self, duration: int = 30):
        """
        Args:
            duration: Benchmark duration in seconds
        """
        self.duration = duration
        self.results = {}

    def measure_baseline(self):
        """Measure baseline CPU/memory without agent"""
        print("📊 Measuring baseline (no agent)...")

        # Get current process
        process = psutil.Process()

        cpu_samples = []
        mem_samples = []

        start = time.time()
        while time.time() - start < self.duration:
            cpu_samples.append(process.cpu_percent(interval=0.1))
            mem_samples.append(process.memory_info().rss / 1024 / 1024)  # MB
            time.sleep(1)

        self.results['baseline'] = {
            'cpu_avg': statistics.mean(cpu_samples),
            'cpu_max': max(cpu_samples),
            'mem_avg': statistics.mean(mem_samples),
            'mem_max': max(mem_samples),
        }

        print(f"   CPU avg: {self.results['baseline']['cpu_avg']:.3f}%")
        print(f"   Memory avg: {self.results['baseline']['mem_avg']:.1f} MB")

    def measure_agent_overhead(self, interval: float = 5.0):
        """Measure overhead with agent running"""
        print(f"\n📊 Measuring agent overhead (interval={interval}s)...")

        def run_agent():
            """Agent process"""
            try:
                with GPUCollector() as collector:
                    while True:
                        collector.collect()
                        time.sleep(interval)
            except KeyboardInterrupt:
                pass

        # Start agent in subprocess
        agent_process = Process(target=run_agent)
        agent_process.start()

        # Give it time to initialize
        time.sleep(2)

        # Monitor the agent process
        try:
            agent_psutil = psutil.Process(agent_process.pid)

            cpu_samples = []
            mem_samples = []

            start = time.time()
            while time.time() - start < self.duration:
                try:
                    cpu_samples.append(agent_psutil.cpu_percent(interval=0.1))
                    mem_samples.append(agent_psutil.memory_info().rss / 1024 / 1024)
                    time.sleep(1)
                except psutil.NoSuchProcess:
                    break

            self.results[f'agent_{interval}s'] = {
                'cpu_avg': statistics.mean(cpu_samples) if cpu_samples else 0,
                'cpu_max': max(cpu_samples) if cpu_samples else 0,
                'mem_avg': statistics.mean(mem_samples) if mem_samples else 0,
                'mem_max': max(mem_samples) if mem_samples else 0,
            }

            print(f"   CPU avg: {self.results[f'agent_{interval}s']['cpu_avg']:.3f}%")
            print(f"   Memory avg: {self.results[f'agent_{interval}s']['mem_avg']:.1f} MB")

        finally:
            # Cleanup
            agent_process.terminate()
            agent_process.join(timeout=5)
            if agent_process.is_alive():
                agent_process.kill()

    def measure_collection_latency(self, num_samples: int = 100):
        """Measure time to collect metrics"""
        print(f"\n📊 Measuring collection latency ({num_samples} samples)...")

        try:
            with GPUCollector() as collector:
                gpu_count = collector.get_gpu_count()

                latencies = []

                for _ in range(num_samples):
                    start = time.perf_counter()
                    collector.collect()
                    elapsed = (time.perf_counter() - start) * 1000  # ms
                    latencies.append(elapsed)

                self.results['latency'] = {
                    'mean_ms': statistics.mean(latencies),
                    'median_ms': statistics.median(latencies),
                    'p95_ms': statistics.quantiles(latencies, n=20)[18],  # 95th percentile
                    'p99_ms': statistics.quantiles(latencies, n=100)[98],  # 99th percentile
                    'max_ms': max(latencies),
                    'gpu_count': gpu_count,
                    'per_gpu_ms': statistics.mean(latencies) / gpu_count if gpu_count > 0 else 0
                }

                print(f"   Mean: {self.results['latency']['mean_ms']:.3f} ms")
                print(f"   Median: {self.results['latency']['median_ms']:.3f} ms")
                print(f"   P95: {self.results['latency']['p95_ms']:.3f} ms")
                print(f"   P99: {self.results['latency']['p99_ms']:.3f} ms")
                print(f"   Per GPU: {self.results['latency']['per_gpu_ms']:.3f} ms")

        except Exception as e:
            print(f"   ❌ Failed: {e}")

    def print_summary(self):
        """Print comprehensive summary"""
        print("\n" + "="*70)
        print(" OVERHEAD BENCHMARK SUMMARY")
        print("="*70)

        if 'baseline' in self.results and 'agent_5.0s' in self.results:
            baseline = self.results['baseline']
            agent = self.results['agent_5.0s']

            cpu_overhead = agent['cpu_avg'] - baseline['cpu_avg']
            mem_overhead = agent['mem_avg'] - baseline['mem_avg']

            print(f"\n{'Metric':<30} {'Baseline':>15} {'With Agent':>15} {'Overhead':>15}")
            print("-"*70)
            print(f"{'CPU Usage (avg)':30} {baseline['cpu_avg']:>14.3f}% {agent['cpu_avg']:>14.3f}% {cpu_overhead:>+14.3f}%")
            print(f"{'CPU Usage (max)':30} {baseline['cpu_max']:>14.3f}% {agent['cpu_max']:>14.3f}%")
            print(f"{'Memory (avg)':30} {baseline['mem_avg']:>14.1f} MB {agent['mem_avg']:>14.1f} MB {mem_overhead:>+14.1f} MB")
            print(f"{'Memory (max)':30} {baseline['mem_max']:>14.1f} MB {agent['mem_max']:>14.1f} MB")

        if 'latency' in self.results:
            lat = self.results['latency']
            print(f"\n{'Collection Latency':<30}")
            print("-"*70)
            print(f"{'Mean latency':30} {lat['mean_ms']:>14.3f} ms")
            print(f"{'P95 latency':30} {lat['p95_ms']:>14.3f} ms")
            print(f"{'P99 latency':30} {lat['p99_ms']:>14.3f} ms")
            print(f"{'GPU count':30} {lat['gpu_count']:>15}")
            print(f"{'Latency per GPU':30} {lat['per_gpu_ms']:>14.3f} ms")

            # Calculate theoretical max sampling rate
            if lat['mean_ms'] > 0:
                max_hz = 1000 / lat['mean_ms']
                print(f"{'Max sampling rate':30} {max_hz:>14.1f} Hz")

        # Pass/Fail criteria
        print("\n" + "-"*70)
        print("PASS/FAIL CRITERIA")
        print("-"*70)

        if 'agent_5.0s' in self.results:
            agent = self.results['agent_5.0s']
            cpu_pass = agent['cpu_avg'] < 1.0  # <1% CPU
            mem_pass = agent['mem_avg'] < 100  # <100 MB

            print(f"{'CPU overhead < 1.0%':40} {'✅ PASS' if cpu_pass else '❌ FAIL':>10}")
            print(f"{'Memory overhead < 100 MB':40} {'✅ PASS' if mem_pass else '❌ FAIL':>10}")

        if 'latency' in self.results:
            lat = self.results['latency']
            latency_pass = lat['p95_ms'] < 5.0  # <5ms p95

            print(f"{'P95 latency < 5.0 ms':40} {'✅ PASS' if latency_pass else '❌ FAIL':>10}")

        print("="*70 + "\n")


def main():
    """Run complete overhead benchmark"""
    print("\n🚀 GPU Energy Agent - Overhead Benchmark")
    print("="*70)
    print("This benchmark measures CPU, memory, and latency overhead.")
    print(f"Benchmark duration: 30 seconds per test")
    print("="*70 + "\n")

    benchmark = OverheadBenchmark(duration=30)

    try:
        # Run benchmarks
        benchmark.measure_baseline()
        benchmark.measure_agent_overhead(interval=5.0)
        benchmark.measure_collection_latency(num_samples=100)

        # Print summary
        benchmark.print_summary()

    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
