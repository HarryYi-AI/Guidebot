"""Developer-facing simulation command."""

from __future__ import annotations

import argparse
import asyncio

from .devices import SimulatedDevice
from .hub import GuidebotHub
from .models import Reading, SensorKind
from .self_evolving import build_default_library
from .simulation import SimulationSuite


async def run_demo() -> None:
    device = SimulatedDevice()
    hub = GuidebotHub(device)
    await hub.start()
    samples = (
        Reading(SensorKind.TEMPERATURE, 29.2, "°C", "simulator"),
        Reading(SensorKind.TOUCH, True, source="simulator"),
        Reading(SensorKind.AIR_QUALITY, 126, "AQI", "simulator"),
    )
    for sample in samples:
        trajectory = await hub.ingest(sample)
        print(f"[{sample.kind}] {trajectory.decision.response or '已记录，无需动作'}")
    await hub.stop()


def run_simulation() -> None:
    report = SimulationSuite().run(build_default_library())
    print(
        f"simulation score={report.score:.4f} "
        f"comfort_error={report.metrics.comfort_error:.3f} "
        f"safety_violations={report.metrics.unsafe_action_count}"
    )


def run_evolve_dry() -> None:
    report = SimulationSuite().run(build_default_library())
    print(
        "dry-run: no skill mutation; "
        f"baseline_score={report.score:.4f}; scenarios={len(report.scenarios)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Guidebot development runtime")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("demo")
    subparsers.add_parser("simulate")
    evolve = subparsers.add_parser("evolve")
    evolve.add_argument("--dry-run", action="store_true", required=True)
    args = parser.parse_args()
    if args.command in (None, "demo"):
        asyncio.run(run_demo())
    elif args.command == "simulate":
        run_simulation()
    elif args.command == "evolve":
        run_evolve_dry()


if __name__ == "__main__":
    main()
