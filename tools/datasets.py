#!/usr/bin/env python3
"""Audit, plan, and explicitly acquire datasets registered for LogAgent.

The default commands are read-only. Downloads require a dataset/profile,
destination, license acknowledgement, and ``--yes``. Conditional or blocked
datasets cannot be fetched by this tool.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Iterable
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = PROJECT_ROOT / "configs" / "datasets.json"
READINESS = {"ready", "conditional", "blocked"}
LICENSE_STATES = {"verified", "unverified", "conflict"}
ADAPTERS = {"huggingface", "zenodo", "github", "manual"}
CHECKSUM_RE = re.compile(r"^(md5|sha256):[0-9a-f]+$")
ID_RE = re.compile(r"^[a-z0-9_]+$")


class RegistryError(RuntimeError):
    """Raised when registry content or a requested action is invalid."""


def load_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RegistryError(f"Registry not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RegistryError(f"Invalid registry JSON: {exc}") from exc


def validate_registry(registry: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if registry.get("schema_version") != 1:
        errors.append("schema_version must be 1")

    datasets = registry.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        return errors + ["datasets must be a non-empty list"]

    seen: set[str] = set()
    for index, dataset in enumerate(datasets):
        prefix = f"datasets[{index}]"
        dataset_id = dataset.get("id")
        if not isinstance(dataset_id, str) or not ID_RE.fullmatch(dataset_id):
            errors.append(f"{prefix}.id must match {ID_RE.pattern}")
            dataset_id = f"index-{index}"
        elif dataset_id in seen:
            errors.append(f"duplicate dataset id: {dataset_id}")
        seen.add(dataset_id)
        label = f"dataset {dataset_id}"

        for key in ("name", "tier", "readiness", "roles", "modalities", "ground_truth", "size", "license", "sources", "profiles"):
            if key not in dataset:
                errors.append(f"{label}: missing {key}")

        if dataset.get("readiness") not in READINESS:
            errors.append(f"{label}: invalid readiness")

        license_info = dataset.get("license", {})
        if license_info.get("status") not in LICENSE_STATES:
            errors.append(f"{label}: invalid license status")
        if not license_info.get("spdx") or not license_info.get("evidence"):
            errors.append(f"{label}: license requires spdx and evidence")

        sources = dataset.get("sources", [])
        if not sources or any(not isinstance(url, str) or not url.startswith("https://") for url in sources):
            errors.append(f"{label}: sources must be non-empty HTTPS URLs")

        profiles = dataset.get("profiles", {})
        if not isinstance(profiles, dict) or not profiles:
            errors.append(f"{label}: profiles must be a non-empty object")
            continue

        for profile_name, profile in profiles.items():
            plabel = f"{label} profile {profile_name}"
            adapter = profile.get("adapter")
            if adapter not in ADAPTERS:
                errors.append(f"{plabel}: invalid adapter")
                continue
            if not profile.get("approx_size"):
                errors.append(f"{plabel}: approx_size is required")
            if "license" in profile:
                profile_license = profile["license"]
                if profile_license.get("status") not in LICENSE_STATES:
                    errors.append(f"{plabel}: invalid profile license status")
                if not profile_license.get("spdx") or not profile_license.get("evidence"):
                    errors.append(f"{plabel}: profile license requires spdx and evidence")
            if adapter == "zenodo":
                if not isinstance(profile.get("record_id"), int):
                    errors.append(f"{plabel}: integer record_id is required")
                if not profile.get("files") and not profile.get("file_name_patterns"):
                    errors.append(f"{plabel}: files or file_name_patterns is required")
            elif adapter == "huggingface":
                if not profile.get("repo_id") or not profile.get("allow_patterns"):
                    errors.append(f"{plabel}: repo_id and allow_patterns are required")
            elif adapter == "github":
                if not str(profile.get("repo", "")).startswith("https://github.com/"):
                    errors.append(f"{plabel}: HTTPS GitHub repo is required")
                if not profile.get("revision"):
                    errors.append(f"{plabel}: revision is required")
            elif adapter == "manual" and not profile.get("instructions"):
                errors.append(f"{plabel}: instructions are required")

            for filename, checksum in profile.get("checksums", {}).items():
                if not filename or not isinstance(checksum, str) or not CHECKSUM_RE.fullmatch(checksum):
                    errors.append(f"{plabel}: invalid checksum for {filename!r}")
            for filename, byte_count in profile.get("expected_bytes", {}).items():
                if not filename or not isinstance(byte_count, int) or byte_count < 0:
                    errors.append(f"{plabel}: invalid expected byte count for {filename!r}")
    return errors


def dataset_by_id(registry: dict[str, Any], dataset_id: str) -> dict[str, Any]:
    for dataset in registry["datasets"]:
        if dataset["id"] == dataset_id:
            return dataset
    available = ", ".join(sorted(d["id"] for d in registry["datasets"]))
    raise RegistryError(f"Unknown dataset {dataset_id!r}. Available: {available}")


def profile_by_name(dataset: dict[str, Any], profile_name: str) -> dict[str, Any]:
    try:
        return dataset["profiles"][profile_name]
    except KeyError as exc:
        available = ", ".join(sorted(dataset["profiles"]))
        raise RegistryError(
            f"Unknown profile {profile_name!r} for {dataset['id']}. Available: {available}"
        ) from exc


def command_catalog(registry: dict[str, Any]) -> None:
    columns = ("ID", "TIER", "STATUS", "LICENSE", "SIZE", "ROLE")
    rows = []
    for dataset in sorted(registry["datasets"], key=lambda item: (item["tier"], item["id"])):
        rows.append(
            (
                dataset["id"],
                str(dataset["tier"]),
                dataset["readiness"].upper(),
                f"{dataset['license']['spdx']}:{dataset['license']['status']}",
                dataset["size"],
                ",".join(dataset["roles"]),
            )
        )
    widths = [max(len(columns[i]), *(len(row[i]) for row in rows)) for i in range(len(columns))]
    print("  ".join(value.ljust(widths[i]) for i, value in enumerate(columns)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


def _license_for_profile(dataset: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    return profile.get("license", dataset["license"])


def command_audit(registry: dict[str, Any]) -> None:
    errors = validate_registry(registry)
    if errors:
        print("REGISTRY AUDIT: FAIL", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        raise RegistryError(f"{len(errors)} registry error(s)")
    ready = sum(d["readiness"] == "ready" for d in registry["datasets"])
    conditional = sum(d["readiness"] == "conditional" for d in registry["datasets"])
    blocked = sum(d["readiness"] == "blocked" for d in registry["datasets"])
    print(
        "REGISTRY AUDIT: PASS "
        f"({len(registry['datasets'])} datasets: {ready} ready, "
        f"{conditional} conditional, {blocked} blocked)"
    )


def command_plan(dataset: dict[str, Any], profile_name: str) -> None:
    profile = profile_by_name(dataset, profile_name)
    license_info = _license_for_profile(dataset, profile)
    fetchable = (
        dataset["readiness"] == "ready"
        and license_info["status"] == "verified"
        and profile["adapter"] != "manual"
    )
    print(f"dataset:       {dataset['id']} — {dataset['name']}")
    print(f"profile:       {profile_name}")
    print(f"readiness:     {dataset['readiness']}")
    print(f"adapter:       {profile['adapter']}")
    print(f"approx size:   {profile['approx_size']}")
    print(f"license:       {license_info['spdx']} ({license_info['status']})")
    print(f"license proof: {license_info['evidence']}")
    print(f"ground truth:  {', '.join(dataset['ground_truth'])}")
    print(f"fetchable:     {'yes' if fetchable else 'no — resolve readiness/license gate'}")
    if profile["adapter"] == "zenodo":
        requested = profile.get("files") or profile.get("file_name_patterns")
        print(f"Zenodo record: https://zenodo.org/records/{profile['record_id']}")
        print("files:         " + ", ".join(requested))
    elif profile["adapter"] == "huggingface":
        print(f"HF repository: https://huggingface.co/datasets/{profile['repo_id']}")
        print("patterns:      " + ", ".join(profile["allow_patterns"]))
    elif profile["adapter"] == "github":
        print(f"repository:    {profile['repo']} @ {profile['revision']}")
    else:
        print(f"instructions:  {profile['instructions']}")
    if dataset.get("limitations"):
        print("limitations:")
        for limitation in dataset["limitations"]:
            print(f"  - {limitation}")
    if fetchable:
        print("command:")
        print(
            f"  python tools/datasets.py fetch {dataset['id']} --profile {profile_name} "
            "--dest data/raw --accept-license --yes"
        )


def _ensure_fetch_allowed(dataset: dict[str, Any], profile: dict[str, Any], args: argparse.Namespace) -> None:
    if dataset["readiness"] != "ready":
        raise RegistryError(
            f"{dataset['id']} is {dataset['readiness']}; resolve its readiness gate before fetching"
        )
    if _license_for_profile(dataset, profile)["status"] != "verified":
        raise RegistryError(f"{dataset['id']} license is not verified")
    if profile["adapter"] == "manual":
        raise RegistryError("This profile is plan-only and has no automatic adapter")
    if not args.accept_license:
        raise RegistryError("Read the upstream terms and pass --accept-license")
    if not args.yes:
        raise RegistryError("Download is disabled without --yes")


def _safe_target(destination: Path, dataset_id: str, profile_name: str) -> Path:
    destination = destination.expanduser().resolve()
    if destination == Path(destination.anchor):
        raise RegistryError("Destination cannot be a filesystem root")
    target = destination / dataset_id / profile_name
    if target.exists():
        raise RegistryError(f"Refusing to overwrite existing target: {target}")
    return target


def _write_provenance(target: Path, payload: dict[str, Any]) -> None:
    (target / ".logagent-source.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _checksum(path: Path, specification: str) -> str:
    algorithm, expected = specification.split(":", 1)
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != expected:
        raise RegistryError(
            f"Checksum mismatch for {path.name}: expected {specification}, got {algorithm}:{actual}"
        )
    return f"{algorithm}:{actual}"


def _download(url: str, destination: Path) -> None:
    partial = destination.with_name(destination.name + ".part")
    request = Request(url, headers={"User-Agent": "LogAgent-dataset-auditor/1"})
    print(f"downloading {destination.name}")
    with urlopen(request) as response, partial.open("xb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    os.replace(partial, destination)


def _select_zenodo_files(metadata: dict[str, Any], profile: dict[str, Any]) -> list[dict[str, Any]]:
    available = {item["key"]: item for item in metadata.get("files", [])}
    requested = profile.get("files", [])
    patterns = profile.get("file_name_patterns", [])
    if requested == ["*"]:
        selected = list(available.values())
    else:
        missing = [name for name in requested if name not in available]
        if missing:
            raise RegistryError(f"Zenodo record is missing requested file(s): {', '.join(missing)}")
        selected = [available[name] for name in requested]
    for pattern in patterns:
        selected.extend(item for name, item in available.items() if fnmatch.fnmatch(name, pattern))
    deduplicated = {item["key"]: item for item in selected}
    if not deduplicated:
        raise RegistryError("No Zenodo files matched the profile")
    return [deduplicated[name] for name in sorted(deduplicated)]


def _fetch_zenodo(dataset: dict[str, Any], profile_name: str, profile: dict[str, Any], target: Path) -> None:
    record_id = profile["record_id"]
    api_url = f"https://zenodo.org/api/records/{record_id}"
    request = Request(api_url, headers={"User-Agent": "LogAgent-dataset-auditor/1"})
    with urlopen(request) as response:
        metadata = json.load(response)
    selected = _select_zenodo_files(metadata, profile)

    expected = profile.get("checksums", {})
    for item in selected:
        registry_checksum = expected.get(item["key"])
        remote_checksum = item.get("checksum")
        if registry_checksum and remote_checksum and registry_checksum != remote_checksum:
            raise RegistryError(
                f"Upstream checksum changed for {item['key']}: "
                f"registry={registry_checksum}, remote={remote_checksum}"
            )

    target.mkdir(parents=True, exist_ok=False)
    resolved: dict[str, str] = {}
    for item in selected:
        filename = item["key"]
        if Path(filename).name != filename or filename in {".", ".."}:
            raise RegistryError(f"Unsafe Zenodo filename: {filename!r}")
        url = item.get("links", {}).get("content") or item.get("links", {}).get("self")
        if not url:
            raise RegistryError(f"No download URL for Zenodo file {filename}")
        destination = target / filename
        _download(url, destination)
        checksum = expected.get(filename) or item.get("checksum")
        if checksum:
            resolved[filename] = _checksum(destination, checksum)

    _write_provenance(
        target,
        {
            "dataset": dataset["id"],
            "profile": profile_name,
            "adapter": "zenodo",
            "record_id": record_id,
            "record_url": f"https://zenodo.org/records/{record_id}",
            "checksums": resolved,
        },
    )


def _fetch_huggingface(dataset: dict[str, Any], profile_name: str, profile: dict[str, Any], target: Path) -> None:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise RegistryError("Install huggingface_hub to fetch Hugging Face profiles") from exc

    info = HfApi().dataset_info(repo_id=profile["repo_id"], revision=profile.get("revision"))
    snapshot_download(
        repo_id=profile["repo_id"],
        repo_type="dataset",
        revision=info.sha,
        allow_patterns=profile["allow_patterns"],
        local_dir=target,
    )
    resolved: dict[str, str] = {}
    for relative_name, checksum in profile.get("checksums", {}).items():
        path = target / relative_name
        if not path.is_file():
            raise RegistryError(f"Expected Hugging Face file is missing: {relative_name}")
        expected_bytes = profile.get("expected_bytes", {}).get(relative_name)
        if expected_bytes is not None and path.stat().st_size != expected_bytes:
            raise RegistryError(
                f"Size mismatch for {relative_name}: expected {expected_bytes}, got {path.stat().st_size}"
            )
        resolved[relative_name] = _checksum(path, checksum)
    _write_provenance(
        target,
        {
            "dataset": dataset["id"],
            "profile": profile_name,
            "adapter": "huggingface",
            "repo_id": profile["repo_id"],
            "revision": info.sha,
            "allow_patterns": profile["allow_patterns"],
            "checksums": resolved,
        },
    )


def _fetch_github(dataset: dict[str, Any], profile_name: str, profile: dict[str, Any], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            profile["revision"],
            profile["repo"],
            str(target),
        ],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(target), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _write_provenance(
        target,
        {
            "dataset": dataset["id"],
            "profile": profile_name,
            "adapter": "github",
            "repo": profile["repo"],
            "requested_revision": profile["revision"],
            "resolved_commit": commit,
        },
    )


def command_fetch(
    dataset: dict[str, Any], profile_name: str, destination: Path, args: argparse.Namespace
) -> None:
    profile = profile_by_name(dataset, profile_name)
    _ensure_fetch_allowed(dataset, profile, args)
    target = _safe_target(destination, dataset["id"], profile_name)
    print(f"target: {target}")
    adapter = profile["adapter"]
    if adapter == "zenodo":
        _fetch_zenodo(dataset, profile_name, profile, target)
    elif adapter == "huggingface":
        _fetch_huggingface(dataset, profile_name, profile, target)
    elif adapter == "github":
        _fetch_github(dataset, profile_name, profile, target)
    else:
        raise RegistryError(f"Unsupported automatic adapter: {adapter}")
    print(f"complete: {target}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("catalog", help="List all registered datasets")
    subparsers.add_parser("audit", help="Validate registry metadata without network access")

    plan = subparsers.add_parser("plan", help="Show one acquisition plan without downloading")
    plan.add_argument("dataset")
    plan.add_argument("--profile", required=True)

    fetch = subparsers.add_parser("fetch", help="Explicitly acquire one ready dataset profile")
    fetch.add_argument("dataset")
    fetch.add_argument("--profile", required=True)
    fetch.add_argument("--dest", type=Path, required=True)
    fetch.add_argument("--accept-license", action="store_true")
    fetch.add_argument("--yes", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        registry = load_registry(args.registry)
        errors = validate_registry(registry)
        if errors and args.command != "audit":
            raise RegistryError("Registry audit failed; run the audit command for details")
        if args.command == "catalog":
            command_catalog(registry)
        elif args.command == "audit":
            command_audit(registry)
        elif args.command == "plan":
            command_plan(dataset_by_id(registry, args.dataset), args.profile)
        elif args.command == "fetch":
            command_fetch(
                dataset_by_id(registry, args.dataset), args.profile, args.dest, args
            )
        return 0
    except (RegistryError, OSError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
