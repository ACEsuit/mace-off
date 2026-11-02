#!/usr/bin/env python3
"""Scan SPICE extxyz files for unusually large reference-force residuals."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from ase.data import covalent_radii
from ase.io import iread
from ase.neighborlist import primitive_neighbor_list


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FILES = (
    ROOT / "data" / "test_large_neut_all.xyz",
)
DEFAULT_REPORT = ROOT / "data" / "force_quality_issues.csv"
ATOM_FORCE_LIMIT = 15.0  # eV/Angstrom
NET_FORCE_LIMIT = 0.1  # eV/Angstrom


def molecular_components(numbers: np.ndarray, positions: np.ndarray) -> list[np.ndarray]:
    """Infer covalent fragments using ASE covalent radii (1.2 radius sum)."""
    parent = list(range(len(numbers)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # ASE uses the per-atom radii additively, giving cutoff_ij = r_i + r_j.
    # Its neighbor list avoids the O(N^2) all-pairs scan for every frame.
    atom_cutoffs = 1.20 * covalent_radii[numbers]
    atom_i, atom_j = primitive_neighbor_list(
        "ij", [False, False, False], np.zeros((3, 3)), positions,
        atom_cutoffs, self_interaction=False,
    )
    for i, j in zip(atom_i, atom_j):
        if i < j:
            left, right = root(int(i)), root(int(j))
            if left != right:
                parent[right] = left
    fragments: dict[int, list[int]] = defaultdict(list)
    for i in range(len(numbers)):
        fragments[root(i)].append(i)
    return [np.asarray(indices, dtype=int) for indices in fragments.values()]


def inspect_file(path: Path) -> tuple[list[dict], dict, Counter]:
    issues = []
    summary = {
        "frames": 0,
        "atomic_force_hits": 0,
        "system_net_force_hits": 0,
        "molecular_net_force_hits": 0,
        "flagged_frames": 0,
    }
    counts_by_type: Counter = Counter()

    for frame_index, atoms in enumerate(iread(str(path), format="extxyz"), start=1):
        forces = np.asarray(atoms.get_forces(), dtype=np.float64)
        if forces.shape != (len(atoms), 3) or not np.isfinite(forces).all():
            raise ValueError(f"{path}: invalid reference forces in frame {frame_index}")
        magnitudes = np.linalg.norm(forces, axis=1)
        max_atom_force = float(magnitudes.max(initial=0.0))
        system_net_force = float(np.linalg.norm(forces.sum(axis=0)))
        fragments = molecular_components(atoms.numbers, atoms.positions)
        fragment_net_forces = [float(np.linalg.norm(forces[idx].sum(axis=0))) for idx in fragments]
        max_molecular_net_force = max(fragment_net_forces, default=0.0)

        atom_hit = max_atom_force > ATOM_FORCE_LIMIT
        system_hit = system_net_force > NET_FORCE_LIMIT
        molecular_hit = max_molecular_net_force > NET_FORCE_LIMIT
        # Only the requested atom-force and whole-system net-force rules trigger
        # exclusion. Fragment net force is reported as a diagnostic because
        # intermolecular interactions can make it nonzero by design.
        flagged = atom_hit or system_hit

        label = str(atoms.info.get("config_type", "<missing>"))
        counts_by_type[label] += 1
        summary["frames"] += 1
        summary["atomic_force_hits"] += int(atom_hit)
        summary["system_net_force_hits"] += int(system_hit)
        summary["molecular_net_force_hits"] += int(molecular_hit)
        summary["flagged_frames"] += int(flagged)

        if flagged:
            issues.append({
                "file": str(path),
                "frame": frame_index,
                "config_type": label,
                "n_atoms": len(atoms),
                "max_atom_force_eV_per_A": max_atom_force,
                "system_net_force_eV_per_A": system_net_force,
                "max_molecular_net_force_eV_per_A": max_molecular_net_force,
                "atomic_force_over_15": atom_hit,
                "system_net_force_over_0p1": system_hit,
                "molecular_net_force_over_0p1": molecular_hit,
            })
        if frame_index % 100000 == 0:
            print(f"{path.name}: scanned {frame_index:,} frames", flush=True)

    return issues, summary, counts_by_type


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path, default=list(DEFAULT_FILES),
                        help="XYZ files to inspect (defaults to the train and test files in data/)")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT,
                        help="CSV path for every flagged frame")
    args = parser.parse_args()

    all_issues = []
    overall = Counter()
    for path in args.files:
        if not path.is_file():
            raise FileNotFoundError(path)
        issues, summary, by_type = inspect_file(path)
        all_issues.extend(issues)
        overall.update(summary)
        print(f"\n{path}: {summary['frames']:,} frames")
        print(f"  max atomic force > {ATOM_FORCE_LIMIT:g} eV/Å: {summary['atomic_force_hits']:,}")
        print(f"  whole-configuration net force > {NET_FORCE_LIMIT:g} eV/Å: "
              f"{summary['system_net_force_hits']:,}")
        print(f"  any molecular-fragment net force > {NET_FORCE_LIMIT:g} eV/Å: "
              f"{summary['molecular_net_force_hits']:,}")
        print(f"  frames flagged by any criterion: {summary['flagged_frames']:,}")
        print("  frames by config_type: " + ", ".join(f"{name}={n:,}" for name, n in by_type.most_common()))

    columns = [
        "file", "frame", "config_type", "n_atoms", "max_atom_force_eV_per_A",
        "system_net_force_eV_per_A", "max_molecular_net_force_eV_per_A",
        "atomic_force_over_15", "system_net_force_over_0p1", "molecular_net_force_over_0p1",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_issues)
    print(f"\nTotal: {overall['frames']:,} frames; {overall['flagged_frames']:,} flagged; "
          f"details: {args.report}")


if __name__ == "__main__":
    main()
