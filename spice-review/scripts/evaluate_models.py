#!/usr/bin/env python3
"""Evaluate the three local MACE-OFF23 models on the SPICE extxyz test set."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.data import atomic_masses, covalent_radii
from ase.io import iread
from ase.neighborlist import primitive_neighbor_list
from mace.calculators import MACECalculator


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEST = ROOT / "data" / "test_large_neut_all.xyz"
MODEL_DIR = ROOT / "models" / "mace_off23"
MODEL_FILES = (
    ("MACE-OFF23(S)", MODEL_DIR / "MACE-OFF23_small.model"),
    ("MACE-OFF23(M)", MODEL_DIR / "MACE-OFF23_medium.model"),
    ("MACE-OFF23(L)", MODEL_DIR / "MACE-OFF23_large.model"),
)
OUTPUT_FIGURE = ROOT / "results" / "fig1.png"
OUTPUT_CSV = ROOT / "results" / "fig1_metrics.csv"
MAX_ATOM_FORCE_EVA = 15.0
MAX_SYSTEM_NET_FORCE_EVA = 0.1


@dataclass(frozen=True)
class Subset:
    name: str
    expected_count: int
    min_atoms: int
    max_atoms: int
    multi_molecule: bool = False


SUBSETS = (
    Subset("PubChem", 33884, 3, 50),
    Subset("DES370K Monomers", 889, 3, 22),
    Subset("DES370K Dimers", 13896, 4, 34, True),
    Subset("Dipeptides", 1025, 26, 60),
    Subset("Solvated Amino Acids", 52, 79, 96, True),
    Subset("Water", 84, 3, 150, True),
    Subset("QMugs", 144, 51, 90),
    Subset("Tripeptides", 898, 30, 69),
)
BY_LABEL = {subset.name.casefold(): subset for subset in SUBSETS}


def molecular_components(numbers: np.ndarray, positions: np.ndarray) -> list[np.ndarray]:
    """Infer covalent components from interatomic distances and ASE radii."""
    parent = list(range(len(numbers)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    cutoffs = 1.20 * covalent_radii[numbers]
    atom_i, atom_j = primitive_neighbor_list(
        "ij", [False, False, False], np.zeros((3, 3)), positions,
        cutoffs, self_interaction=False,
    )
    for i, j in zip(atom_i, atom_j):
        if i < j:
            a, b = root(int(i)), root(int(j))
            if a != b:
                parent[b] = a
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(numbers)):
        groups[root(i)].append(i)
    return [np.asarray(indices, dtype=int) for indices in groups.values()]


def intermolecular_force(numbers: np.ndarray, positions: np.ndarray,
                         forces: np.ndarray) -> np.ndarray:
    """Project each molecule's rigid translation/rotation into atomic forces."""
    result = np.zeros_like(forces, dtype=np.float64)
    for indices in molecular_components(numbers, positions):
        masses = atomic_masses[numbers[indices]]
        com = np.average(positions[indices], axis=0, weights=masses)
        relative = positions[indices] - com
        molecular_forces = forces[indices]
        translation = (masses[:, None] / masses.sum()) * molecular_forces.sum(axis=0)

        inertia = np.zeros((3, 3), dtype=np.float64)
        for mass, r in zip(masses, relative):
            inertia += mass * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
        # Torque convention here is r x f, equivalent to the supplied
        # f x r / r x I^-1 T formulation after accounting for its sign.
        torque = np.sum(np.cross(relative, molecular_forces), axis=0)
        omega = np.linalg.pinv(inertia, rcond=1e-12) @ torque
        rotation = masses[:, None] * np.cross(omega[None, :], relative)
        result[indices] = translation + rotation
    return result


def new_accumulator() -> dict:
    return {
        "n_configs": 0,
        "energy_sq": 0.0,
        "force_sq": 0.0,
        "force_components": 0,
        "inter_sq": 0.0,
        "inter_components": 0,
        "dft_inter_sq": 0.0,
    }


def evaluate(test_path: Path, device: str, limit: int | None = None):
    calculators = [(name, MACECalculator(model_paths=str(path), device=device,
                                         default_dtype="float64"))
                   for name, path in MODEL_FILES]
    totals = {subset.name: {name: new_accumulator() for name, _ in MODEL_FILES}
              for subset in SUBSETS}
    seen = defaultdict(int)
    input_counts = defaultdict(int)
    excluded = defaultdict(lambda: {"atomic_force": 0, "system_net_force": 0, "either": 0})
    n_read = 0

    for atoms in iread(str(test_path), format="extxyz"):
        if limit is not None and n_read >= limit:
            break
        n_read += 1
        label = str(atoms.info.get("config_type", "")).casefold()
        subset = BY_LABEL.get(label)
        if subset is None:
            raise ValueError(f"Unknown config_type at frame {n_read}: {label!r}")

        numbers = atoms.numbers
        positions = atoms.positions
        n_atoms = len(atoms)
        input_counts[subset.name] += 1
        if not subset.min_atoms <= n_atoms <= subset.max_atoms:
            raise ValueError(f"{subset.name}: frame {n_read} has {n_atoms} atoms, outside "
                             f"{subset.min_atoms}–{subset.max_atoms}")
        reference_energy = float(atoms.get_potential_energy())  # eV
        reference_forces = np.asarray(atoms.get_forces(), dtype=np.float64)  # eV/Å
        atom_force_bad = np.linalg.norm(reference_forces, axis=1).max() > MAX_ATOM_FORCE_EVA
        system_net_force_bad = (np.linalg.norm(reference_forces.sum(axis=0))
                                > MAX_SYSTEM_NET_FORCE_EVA)
        excluded[subset.name]["atomic_force"] += int(atom_force_bad)
        excluded[subset.name]["system_net_force"] += int(system_net_force_bad)
        if atom_force_bad or system_net_force_bad:
            excluded[subset.name]["either"] += 1
            continue

        reference_inter = intermolecular_force(numbers, positions, reference_forces)
        seen[subset.name] += 1

        for model_name, calculator in calculators:
            prediction = atoms.copy()
            prediction.calc = calculator
            predicted_energy = float(prediction.get_potential_energy())
            predicted_forces = np.asarray(prediction.get_forces(), dtype=np.float64)
            predicted_inter = intermolecular_force(numbers, positions, predicted_forces)

            row = totals[subset.name][model_name]
            row["n_configs"] += 1
            row["energy_sq"] += ((reference_energy - predicted_energy) / n_atoms * 1000.0) ** 2
            force_diff = (reference_forces - predicted_forces) * 1000.0
            row["force_sq"] += float(np.sum(force_diff ** 2))
            row["force_components"] += force_diff.size
            inter_diff = (reference_inter - predicted_inter) * 1000.0
            row["inter_sq"] += float(np.sum(inter_diff ** 2))
            row["inter_components"] += inter_diff.size
            row["dft_inter_sq"] += float(np.sum((reference_inter * 1000.0) ** 2))

        if n_read % 100 == 0:
            print(f"Processed {n_read:,} configurations", flush=True)

    return totals, dict(seen), dict(input_counts), {k: dict(v) for k, v in excluded.items()}, n_read


def build_rows(totals: dict, seen: dict, input_counts: dict, excluded: dict) -> list[dict]:
    rows = []
    for subset in SUBSETS:
        count = seen.get(subset.name, 0)
        for model_name, _ in MODEL_FILES:
            row = totals[subset.name][model_name]
            if not count:
                energy_rmse = force_rmse = inter_rmse = dft_inter_rms = np.nan
            else:
                energy_rmse = np.sqrt(row["energy_sq"] / count)
                force_rmse = np.sqrt(row["force_sq"] / row["force_components"])
                inter_rmse = np.sqrt(row["inter_sq"] / row["inter_components"])
                dft_inter_rms = np.sqrt(row["dft_inter_sq"] / row["inter_components"])
            rows.append({
                "subset": subset.name,
                "n_test": count,
                "n_input": input_counts.get(subset.name, 0),
                "n_excluded": excluded.get(subset.name, {}).get("either", 0),
                "n_excluded_atom_force": excluded.get(subset.name, {}).get("atomic_force", 0),
                "n_excluded_system_net_force": excluded.get(subset.name, {}).get("system_net_force", 0),
                "expected_n_test": subset.expected_count,
                "model": model_name,
                "energy_rmse_mev_per_atom": energy_rmse,
                "force_rmse_mev_per_angstrom": force_rmse,
                "inter_force_rmse_mev_per_angstrom": inter_rmse,
                "dft_inter_force_rms_mev_per_angstrom": dft_inter_rms,
            })
    return rows


def save_outputs(rows: list[dict], seen: dict, csv_path: Path, figure_path: Path) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    x = np.arange(len(SUBSETS), dtype=float)
    width = 0.22
    colors = ("#4C78A8", "#F58518", "#54A24B")
    metrics = (
        ("energy_rmse_mev_per_atom", "Energy RMSE (meV/atom)"),
        ("force_rmse_mev_per_angstrom", "Force RMSE (meV/Å)"),
        ("inter_force_rmse_mev_per_angstrom", "Inter Force RMSE (meV/Å)"),
    )
    lookup = {(row["subset"], row["model"]): row for row in rows}
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True, constrained_layout=True)
    for ax, (key, ylabel) in zip(axes, metrics):
        for model_i, (model_name, _) in enumerate(MODEL_FILES):
            values = [lookup[(spec.name, model_name)][key] for spec in SUBSETS]
            ax.bar(x + (model_i - 1) * width, values, width, color=colors[model_i], label=model_name)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    ref_values = [lookup[(spec.name, MODEL_FILES[0][0])]
                  ["dft_inter_force_rms_mev_per_angstrom"] if spec.multi_molecule else np.nan
                  for spec in SUBSETS]
    axes[2].bar(x + 1.5 * width, ref_values, width * 0.9, color="#D0D0D0",
                edgecolor="#555555", hatch="////", label="DFT Inter Force RMS")
    axes[0].legend(ncol=3, frameon=False, loc="upper right")
    axes[2].legend(ncol=4, frameon=False, loc="upper right")
    axes[-1].set_xticks(x)
    tick_labels = [f"{s.name}\n(n={seen.get(s.name, 0):,})" for s in SUBSETS]
    axes[-1].set_xticklabels(tick_labels, rotation=15, ha="right")
    fig.suptitle("MACE-OFF23 performance on SPICE test configurations")
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xyz", type=Path, default=DEFAULT_TEST, help="extxyz test set; train file is not used")
    parser.add_argument("--device", default="cpu", help="MACE device, e.g. cpu or cuda")
    parser.add_argument("--limit", type=int, help="process only the first N frames for a quick inference check")
    parser.add_argument("--csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--figure", type=Path, default=OUTPUT_FIGURE)
    args = parser.parse_args()

    missing = [str(path) for _, path in MODEL_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing MACE model files: " + ", ".join(missing))
    totals, seen, input_counts, excluded, processed = evaluate(args.xyz, args.device, args.limit)
    for subset in SUBSETS:
        n_input = input_counts.get(subset.name, 0)
        n_skipped = excluded.get(subset.name, {}).get("either", 0)
        if n_input:
            print(f"{subset.name}: read {n_input:,}, excluded {n_skipped:,}, "
                  f"evaluated {seen.get(subset.name, 0):,} (reference target {subset.expected_count:,})")
    rows = build_rows(totals, seen, input_counts, excluded)
    save_outputs(rows, seen, args.csv, args.figure)
    print(f"Processed {processed:,} frames; metrics: {args.csv}; figure: {args.figure}")


if __name__ == "__main__":
    main()
