#!/usr/bin/env python3
"""Run one user-selected HW6 correctness stage on Perlmutter.

Load cpu then python; export CC=cc and CXX=CC before build. Use an external,
new --work directory. Run transfers/small/image inside your own salloc session.
This helper never allocates nodes, submits batch jobs, or runs a performance grid.
"""
import argparse
import filecmp
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[2]
SOURCES = ("mpi_2dmesh.cpp", "mpi_2dmesh.hpp", "CMakeLists.txt",
           "scripts/validation/check_mpi.cpp", "scripts/validation/part2.py")
ENV_KEYS = ("LOADEDMODULES", "PE_ENV", "CRAYPE_VERSION", "CRAY_MPICH_VERSION",
            "MPICH_GPU_SUPPORT_ENABLED", "CC", "CXX", "SLURM_JOB_ID",
            "SLURM_JOB_NODELIST", "SLURM_JOB_NUM_NODES")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def source_hashes():
    return {name: digest(REPO / name) for name in SOURCES}


def run(command, log):
    command = [str(arg) for arg in command]
    rendered = "$ " + " ".join(shlex.quote(arg) for arg in command)
    print(rendered, flush=True)
    with log.open("x") as stream:
        stream.write(rendered + "\n")
        stream.flush()
        with subprocess.Popen(command, cwd=REPO, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, errors="replace") as process:
            output = []
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
                output.append(line)
            code = process.wait()
        stream.write(f"exit_status={code}\n")
    if code:
        raise RuntimeError(f"Command exited {code}; inspect {log}")
    return "".join(output)


def outside_repo(path):
    path = path.expanduser().resolve()
    if path == REPO or REPO in path.parents:
        raise ValueError("--work must be outside the harness checkout")
    return path


def environment():
    return {key: os.environ.get(key, "") for key in ENV_KEYS}


def build(work):
    compiler = shlex.split(os.environ.get("CXX", ""))
    if not compiler or not shutil.which(compiler[0]):
        raise ValueError("Load the CPU modules and export CXX=CC before building")
    work.mkdir(parents=True, exist_ok=False)
    (work / "results").mkdir()
    initial_sources = source_hashes()
    (work / "environment.json").write_text(json.dumps(environment(), indent=2) + "\n")
    run(["git", "rev-parse", "HEAD"], work / "revision.log")
    run(["git", "status", "--short"], work / "status.log")
    run(["git", "diff", "HEAD", "--"], work / "source-diff.log")
    run(["uname", "-a"], work / "platform.log")
    run(compiler + ["--version"], work / "compiler.log")
    run(["cmake", "--version"], work / "cmake-version.log")
    run(["cmake", "-S", REPO, "-B", work / "build"], work / "configure.log")
    run(["cmake", "--build", work / "build", "--verbose"], work / "build.log")
    # This separate correctness executable does not change CMake or harness flags.
    run(compiler + ["-O3", "-DNDEBUG", str(REPO / "scripts/validation/check_mpi.cpp"),
                    "-o", str(work / "check_mpi")], work / "build-check.log")
    if initial_sources != source_hashes():
        raise RuntimeError("Sources changed during the build; use a fresh work directory")
    binaries = {"build/mpi_2dmesh": digest(work / "build/mpi_2dmesh"),
                "check_mpi": digest(work / "check_mpi")}
    (work / "build-state.json").write_text(json.dumps(
        {"sources": initial_sources, "binaries": binaries}, indent=2) + "\n")
    print(f"BUILD COMPLETE: {work}\nNo MPI tests have run yet.")


def verify_build(work):
    state = json.loads((work / "build-state.json").read_text())
    if state["sources"] != source_hashes():
        raise ValueError("Checkout differs from the saved build; build into a new --work directory")
    for name, expected in state["binaries"].items():
        if digest(work / name) != expected:
            raise ValueError(f"Binary changed since build: {name}")


def placement(output, ranks, nodes, production=False):
    expression = (r"Hello world, I'm rank (\d+) of (\d+) total ranks running on <([^>]+)>"
                  if production else r"Validation rank (\d+) of (\d+) on <([^>]+)>")
    matches = re.findall(expression, output)
    if (len(matches) != ranks or {int(r) for r, _, _ in matches} != set(range(ranks))
            or any(int(total) != ranks for _, total, _ in matches)):
        raise RuntimeError("Missing, duplicate, or inconsistent rank/hostname lines; inspect the log")
    hosts = sorted({host for _, _, host in matches})
    if len(hosts) != nodes:
        raise RuntimeError(f"Expected {nodes} distinct nodes; observed {hosts}")
    print(f"PLACEMENT OK: {ranks} ranks on {nodes} nodes: {', '.join(hosts)}")


def timing(output):
    values = re.findall(r"(Scatter|Sobel|Gather) time:\s+([0-9]+\.[0-9]+) \(ms\)", output)
    if (len(values) != 3 or {phase for phase, _ in values} != {"Scatter", "Sobel", "Gather"}
            or any(float(value) <= 0 for _, value in values)):
        raise RuntimeError("Expected three positive phase times in milliseconds; inspect timing output")
    print("TIMING OUTPUT OK (correctness run, not retained performance data)")


def validate(args, work):
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("Run this stage inside a CPU salloc allocation")
    verify_build(work)
    ranks, nodes = (2, 1) if args.stage == "transfers" else (args.ranks, args.nodes)
    allocated = os.environ.get("SLURM_JOB_NUM_NODES")
    if allocated and nodes > int(allocated):
        raise ValueError(f"Requested {nodes} nodes but allocation has {allocated}")
    result = Path(tempfile.mkdtemp(prefix=f"{args.stage}-p{ranks}-n{nodes}-", dir=work / "results"))
    print(f"Evidence directory: {result}", flush=True)
    (result / "environment.json").write_text(json.dumps(environment(), indent=2) + "\n")
    launch = ["srun", f"--nodes={nodes}", f"--ntasks={ranks}", "--cpu-bind=cores",
              "--kill-on-bad-exit=1", "--time=00:05:00"]
    checker = [work / "check_mpi", args.stage]
    if args.stage == "image":
        filename, width, height = (("zebra-gray-int8", 3556, 2573) if args.size == "original"
                                   else ("zebra-gray-int8-4x", 7112, 5146))
        input_path = REPO / "data" / filename
        reference = result / "serial-reference.raw"
        if input_path.stat().st_size != width * height:
            raise ValueError("Supplied input size differs from its declared dimensions")
        if max(len(str(input_path).encode()), len(str(reference).encode())) >= 256:
            raise ValueError("Use shorter checkout/work paths: starter filename buffers hold 256 bytes")
        (result / "input.sha256").write_text(digest(input_path) + "  " + str(input_path) + "\n")
        checker += [input_path, width, height, reference]
    output = run(launch + checker, result / "check.log")
    placement(output, ranks, nodes)
    if f"PASS: MPI validation {args.stage}; failures=0" not in output:
        raise RuntimeError("Checker exited without its final PASS marker")
    if args.stage == "image":
        if reference.stat().st_size != width * height:
            raise RuntimeError("Serial reference output has the wrong byte count")
        for decomp in (1, 2, 3):
            destination = result / f"production-g{decomp}.raw"
            if len(str(destination).encode()) >= 256:
                raise ValueError("Output path exceeds the starter filename buffer")
            output = run(launch + [work / "build/mpi_2dmesh", "-a", 1, "-g", decomp,
                                   "-i", input_path, "-x", width, "-y", height,
                                   "-o", destination], result / f"production-g{decomp}.log")
            placement(output, ranks, nodes, production=True)
            timing(output)
            if not filecmp.cmp(reference, destination, shallow=False):
                raise RuntimeError(f"Byte output differs from serial reference: {destination}")
            print(f"BYTE COMPARISON OK: g={decomp}; full float comparisons are in check.log")
    (result / "PASS.txt").write_text("Correctness stage passed. Not retained performance data.\n")
    print(f"STAGE PASSED: {result}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    for name in ("build", "transfers", "small", "image"):
        command = commands.add_parser(name)
        command.add_argument("--work", required=True, type=Path,
                             help="external build/evidence directory; must be new for build")
        if name in ("small", "image"):
            command.add_argument("--ranks", type=int, choices=(4, 9), required=True)
            command.add_argument("--nodes", type=int, choices=(1, 4), default=1)
        if name == "image":
            command.add_argument("--size", choices=("original", "4x"), default="original")
    args = parser.parse_args()
    work = outside_repo(args.work)
    if args.stage == "build":
        build(work)
    else:
        validate(args, work)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
