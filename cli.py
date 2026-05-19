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
AluminatAI CLI entry point.

When installed via `pip install aluminatiai`, the `aluminatiai` command
runs this module.  sys.path is patched first so the bare-import modules
(collector, config, uploader, etc.) resolve against the installed package
directory regardless of the working directory.

The --config/-c flag is pre-parsed here (before any other imports) so that
config.py's _load_config_file() sees the ALUMINATAI_CONFIG env var when it
runs at import time.
"""
import argparse
import os
import sys


def main() -> None:
    # Insert the package directory at the front of sys.path so that
    # bare imports like `from collector import GPUCollector` resolve to
    # the modules installed alongside this file (site-packages/aluminatiai/).
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    # ── Pre-parse subcommand + --config before any module imports ─────────
    # We pre-parse the first positional argument as a subcommand so that
    # `aluminatiai benchmark ...` dispatches to benchmark.py without loading
    # the full agent stack.  Unknown args are forwarded to the sub-handler.
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("subcommand", nargs="?", default="run",
                      choices=["run", "benchmark", "optimize", "ab", "demo", "report",
                               "carbon-schedule", "query", "recommend", "train", "swarm"])
    _pre.add_argument("--config", "-c", default=None,
                      help="Path to JSON/YAML config file")
    _known, _rest = _pre.parse_known_args()

    if _known.config:
        os.environ["ALUMINATAI_CONFIG"] = _known.config

    if _known.subcommand == "benchmark":
        from benchmark import make_parser, run_benchmark  # noqa: PLC0415
        sys.exit(run_benchmark(make_parser().parse_args(_rest)))

    if _known.subcommand == "optimize":
        from optimize import make_parser as opt_parser, run_optimize  # noqa: PLC0415
        sys.exit(run_optimize(opt_parser().parse_args(_rest)))

    if _known.subcommand == "ab":
        from ab import make_parser as ab_parser, run_ab  # noqa: PLC0415
        sys.exit(run_ab(ab_parser().parse_args(_rest)))

    if _known.subcommand == "demo":
        from demos.investor_demo import make_parser as demo_parser, run_demo  # noqa: PLC0415
        sys.exit(run_demo(demo_parser().parse_args(_rest)))

    if _known.subcommand == "report":
        from reports.chargeback import make_parser as rpt_parser, run_report  # noqa: PLC0415
        sys.exit(run_report(rpt_parser().parse_args(_rest)))

    if _known.subcommand == "carbon-schedule":
        from efficiency.carbon_scheduler import make_parser as cs_parser, run_carbon_schedule  # noqa: PLC0415
        sys.exit(run_carbon_schedule(cs_parser().parse_args(_rest)))

    if _known.subcommand == "query":
        from storage.tsdb import make_parser as q_parser, run_query  # noqa: PLC0415
        sys.exit(run_query(q_parser().parse_args(_rest)))

    if _known.subcommand == "recommend":
        from recommend import make_parser as rec_parser, run_recommend  # noqa: PLC0415
        sys.exit(run_recommend(rec_parser().parse_args(_rest)))

    if _known.subcommand == "train":
        sys.argv = ["aluminatiai-train"] + _rest
        from finetune.greentune import main as train_main  # noqa: PLC0415
        sys.exit(train_main())

    if _known.subcommand == "swarm":
        sys.argv = ["aluminatiai-swarm"] + _rest
        from finetune.greentune_swarm import main as swarm_main  # noqa: PLC0415
        sys.exit(swarm_main())

    # Default: delegate to agent.main() — it owns the full argparse + run loop.
    from agent import main as _main  # noqa: PLC0415
    sys.exit(_main())


if __name__ == "__main__":
    main()
