#!/usr/bin/env python3
"""Replicate and verify the finite-front Solow--Polasky examples.

This script reproduces the two numerical examples in the report:

1. A dense connected front f2 = 1 - f1^2 with n=70 and k=10.
2. A disconnected ZDT3 Pareto front sample with n=100 and k=20.

It also performs two verification steps:

A. A beta sanity check for the ZDT3 example, comparing beta=1 and beta=2.
B. Several small brute-force checks comparing the dynamic-programming solution
   against complete enumeration on small ordered line instances.

The dynamic program uses the finite-line formula for Solow--Polasky diversity
under the exponential kernel on an ordered l1 front:

    D_beta(S) = 1 + sum_r tanh(beta * gap_r / 2),

where gap_r are consecutive distances in the induced ordered line coordinate.
The script uses only the Python standard library.
"""

from __future__ import annotations

import csv
import itertools
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


Point = Tuple[float, float]
IndexTuple = Tuple[int, ...]  # zero-based internally


@dataclass(frozen=True)
class SelectionResult:
    """Container for one ordered-line subset-selection result."""

    indices_1based: List[int]
    selected_points: List[Point]
    selected_coordinates: List[float]
    gaps: List[float]
    diversity: float
    beta: float
    target_gap: float
    max_abs_gap_deviation: float


def sp_gap_contribution(delta: float, beta: float) -> float:
    """Contribution of one consecutive line gap to SP diversity."""
    return math.tanh(beta * delta / 2.0)


def diversity_from_indices(coords: Sequence[float], indices: Sequence[int], beta: float) -> float:
    """Evaluate D_beta for one ordered subset, using zero-based indices."""
    if not indices:
        raise ValueError("indices must be nonempty")
    selected = [coords[i] for i in indices]
    return 1.0 + sum(
        sp_gap_contribution(selected[i + 1] - selected[i], beta)
        for i in range(len(selected) - 1)
    )


def induced_l1_coordinate(point: Point) -> float:
    """Line coordinate for a biobjective front with f1 increasing and f2 decreasing."""
    x, y = point
    return x - y


def _result_from_indices(
    coords: Sequence[float], indices: Sequence[int], beta: float, points: Sequence[Point] | None = None
) -> SelectionResult:
    """Build a SelectionResult from zero-based indices."""
    selected_coordinates = [coords[i] for i in indices]
    gaps = [selected_coordinates[i + 1] - selected_coordinates[i] for i in range(len(indices) - 1)]
    target_gap = (coords[-1] - coords[0]) / (len(indices) - 1) if len(indices) > 1 else 0.0
    max_dev = max((abs(g - target_gap) for g in gaps), default=0.0)
    selected_points = [points[i] for i in indices] if points is not None else []
    return SelectionResult(
        indices_1based=[i + 1 for i in indices],
        selected_points=selected_points,
        selected_coordinates=selected_coordinates,
        gaps=gaps,
        diversity=diversity_from_indices(coords, indices, beta),
        beta=beta,
        target_gap=target_gap,
        max_abs_gap_deviation=max_dev,
    )


def select_ordered_line_subset(coords: Sequence[float], k: int, beta: float = 1.0) -> SelectionResult:
    """Select k points from ordered line coordinates by dynamic programming.

    The objective is

        1 + sum tanh(beta * consecutive_gap / 2).

    Endpoints are not forced. Since the gap contribution is increasing, the
    first and last candidate are selected automatically in the examples below.
    """
    n = len(coords)
    if not (1 <= k <= n):
        raise ValueError(f"need 1 <= k <= n, got k={k}, n={n}")
    if any(coords[i + 1] < coords[i] for i in range(n - 1)):
        raise ValueError("coordinates must be sorted in nondecreasing order")

    neg_inf = float("-inf")
    dp = [[neg_inf] * n for _ in range(k + 1)]
    parent = [[-1] * n for _ in range(k + 1)]

    for j in range(n):
        dp[1][j] = 0.0

    for r in range(2, k + 1):
        for j in range(r - 1, n):
            best_value = neg_inf
            best_i = -1
            for i in range(r - 2, j):
                value = dp[r - 1][i] + sp_gap_contribution(coords[j] - coords[i], beta)
                if value > best_value:
                    best_value = value
                    best_i = i
            dp[r][j] = best_value
            parent[r][j] = best_i

    end = max(range(n), key=lambda j: dp[k][j])
    indices: List[int] = []
    r = k
    j = end
    while r >= 1:
        indices.append(j)
        j = parent[r][j]
        r -= 1
    indices.reverse()

    return _result_from_indices(coords, indices, beta)


def brute_force_ordered_line_subset(
    coords: Sequence[float], k: int, beta: float = 1.0, tolerance: float = 1e-12
) -> Tuple[float, List[IndexTuple]]:
    """Solve a small ordered-line instance by complete enumeration.

    Returns the optimal diversity value and all zero-based optimal index tuples
    within the requested tolerance. This is intended only for small n.
    """
    n = len(coords)
    if not (1 <= k <= n):
        raise ValueError(f"need 1 <= k <= n, got k={k}, n={n}")
    if any(coords[i + 1] < coords[i] for i in range(n - 1)):
        raise ValueError("coordinates must be sorted in nondecreasing order")

    best_value = float("-inf")
    best_indices: List[IndexTuple] = []
    for combo in itertools.combinations(range(n), k):
        value = diversity_from_indices(coords, combo, beta)
        if value > best_value + tolerance:
            best_value = value
            best_indices = [combo]
        elif abs(value - best_value) <= tolerance:
            best_indices.append(combo)
    return best_value, best_indices


def dense_front_points(n: int = 70) -> List[Point]:
    """Candidates on f2 = 1 - f1^2."""
    return [(i / (n - 1), 1.0 - (i / (n - 1)) ** 2) for i in range(n)]


def zdt3_f2(x: float) -> float:
    """ZDT3 Pareto-front objective value."""
    return 1.0 - math.sqrt(x) - x * math.sin(10.0 * math.pi * x)


def zdt3_front_points(points_per_component: int = 20) -> List[Point]:
    """Candidates on the five ZDT3 Pareto-optimal components.

    The intervals are given with enough digits to make the component boundary
    objective values agree to plotting/numerical precision.
    """
    intervals = [
        (0.0, 0.0830015349269),
        (0.1822287280294, 0.2577623633878),
        (0.4093136748087, 0.4538821040888),
        (0.6183967944393, 0.6525117038047),
        (0.8233317983266, 0.8518328654364),
    ]
    points: List[Point] = []
    for a, b in intervals:
        for j in range(points_per_component):
            x = a + (b - a) * j / (points_per_component - 1)
            points.append((x, zdt3_f2(x)))
    return points


def solve_example(points: List[Point], k: int, beta: float = 1.0) -> SelectionResult:
    coords = [induced_l1_coordinate(p) for p in points]
    result = select_ordered_line_subset(coords, k=k, beta=beta)
    return _result_from_indices(coords, [i - 1 for i in result.indices_1based], beta, points)


def print_result(name: str, result: SelectionResult) -> None:
    print(f"\n{name}")
    print("=" * len(name))
    print(f"beta = {result.beta:g}")
    print("selected indices:", result.indices_1based)
    print(f"D_{result.beta:g} = {result.diversity:.12f}")
    print(f"target gap = {result.target_gap:.12f}")
    print(f"max |gap-target| = {result.max_abs_gap_deviation:.12f}")
    print("gaps:")
    print(", ".join(f"{g:.12f}" for g in result.gaps))
    if result.selected_points:
        print("selected coordinates (f1, f2, induced_coordinate):")
        for (x, y), s in zip(result.selected_points, result.selected_coordinates):
            print(f"  ({x:.12f}, {y:.12f}, {s:.12f})")


def write_outputs(results: Sequence[Tuple[str, SelectionResult]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, result in results:
        safe = name.lower().replace(" ", "_").replace("-", "_").replace("=", "")
        csv_path = out_dir / f"{safe}_selected_points.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["index_1based", "f1", "f2", "induced_coordinate", "beta"])
            for idx, (x, y), s in zip(
                result.indices_1based, result.selected_points, result.selected_coordinates
            ):
                writer.writerow([idx, f"{x:.15g}", f"{y:.15g}", f"{s:.15g}", result.beta])

        tex_path = out_dir / f"{safe}_selected_coordinates.tex"
        with tex_path.open("w") as handle:
            for x, y in result.selected_points:
                handle.write(f"({x:.9f}, {y: .9f})\n")


def assert_close(actual: float, expected: float, tolerance: float = 1e-10) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"expected {expected:.15f}, got {actual:.15f}")


def run_report_value_checks(dense_result: SelectionResult, zdt3_beta1: SelectionResult, zdt3_beta2: SelectionResult) -> None:
    """Check the report's hard-coded selections and numerical values."""
    expected_dense = [1, 14, 24, 32, 40, 47, 53, 59, 65, 70]
    expected_zdt3 = [1, 4, 10, 20, 23, 28, 32, 40, 41, 45, 50, 60, 61, 65, 70, 80, 81, 85, 90, 100]

    if dense_result.indices_1based != expected_dense:
        raise AssertionError(f"dense-front indices changed: {dense_result.indices_1based}")
    if zdt3_beta1.indices_1based != expected_zdt3:
        raise AssertionError(f"ZDT3 beta=1 indices changed: {zdt3_beta1.indices_1based}")
    if zdt3_beta2.indices_1based != expected_zdt3:
        raise AssertionError(f"ZDT3 beta=2 indices changed: {zdt3_beta2.indices_1based}")

    assert_close(dense_result.diversity, 1.995877982033559)
    assert_close(zdt3_beta1.diversity, 2.310417410238659)
    assert_close(zdt3_beta2.diversity, 3.607843892374750)

    print("\nReport-value checks")
    print("===================")
    print("OK: dense-front beta=1 indices and diversity match the report values.")
    print("OK: ZDT3 beta=1 indices and diversity match the report values.")
    print("OK: ZDT3 beta=2 has the same optimal index set as beta=1.")


def run_bruteforce_verification() -> None:
    """Compare the DP with brute force on small ordered-line instances."""
    rng = random.Random(1729)

    cases: List[Tuple[str, List[float], int, float]] = []

    # A hand-made uneven line instance.
    cases.append(("toy uneven line", [0.0, 0.05, 0.11, 0.19, 0.34, 0.56, 0.72, 1.00], 4, 1.0))
    cases.append(("toy uneven line, beta=2", [0.0, 0.05, 0.11, 0.19, 0.34, 0.56, 0.72, 1.00], 4, 2.0))

    # A small connected-front instance: n=12, k=5.
    small_dense = [induced_l1_coordinate(p) for p in dense_front_points(n=12)]
    cases.append(("small dense front", small_dense, 5, 1.0))
    cases.append(("small dense front, beta=2", small_dense, 5, 2.0))

    # A small ZDT3 instance: 3 candidates per component, n=15, k=6.
    small_zdt3 = [induced_l1_coordinate(p) for p in zdt3_front_points(points_per_component=3)]
    cases.append(("small ZDT3", small_zdt3, 6, 1.0))
    cases.append(("small ZDT3, beta=2", small_zdt3, 6, 2.0))

    # A few seeded random sorted line instances.
    for trial in range(1, 5):
        values = sorted(rng.random() for _ in range(10))
        values[0] = 0.0
        values[-1] = 1.0
        k = 4 if trial % 2 else 5
        beta = 0.8 if trial <= 2 else 2.0
        cases.append((f"seeded random line {trial}", values, k, beta))

    print("\nSmall brute-force verification")
    print("==============================")
    for name, coords, k, beta in cases:
        dp_result = select_ordered_line_subset(coords, k=k, beta=beta)
        dp_zero_based = tuple(i - 1 for i in dp_result.indices_1based)
        brute_value, brute_optima = brute_force_ordered_line_subset(coords, k=k, beta=beta)
        if abs(dp_result.diversity - brute_value) > 1e-11:
            raise AssertionError(
                f"{name}: DP value {dp_result.diversity:.15f} != brute value {brute_value:.15f}"
            )
        if dp_zero_based not in brute_optima:
            raise AssertionError(
                f"{name}: DP indices {dp_result.indices_1based} not among brute-force optima"
            )
        print(
            f"OK: {name:27s} n={len(coords):2d}, k={k}, beta={beta:g}, "
            f"D={dp_result.diversity:.12f}, optima={len(brute_optima)}"
        )


def main() -> None:
    dense_points = dense_front_points(n=70)
    dense_result = solve_example(dense_points, k=10, beta=1.0)

    zdt3_points = zdt3_front_points(points_per_component=20)
    zdt3_beta1 = solve_example(zdt3_points, k=20, beta=1.0)
    zdt3_beta2 = solve_example(zdt3_points, k=20, beta=2.0)

    print_result("Dense connected front, n=70, k=10", dense_result)
    print_result("Disconnected ZDT3 front, n=100, k=20, beta=1", zdt3_beta1)
    print_result("Disconnected ZDT3 front, n=100, k=20, beta=2", zdt3_beta2)

    print("\nBeta sanity check for ZDT3")
    print("==========================")
    print("Same optimal indices for beta=1 and beta=2:", zdt3_beta1.indices_1based == zdt3_beta2.indices_1based)

    run_report_value_checks(dense_result, zdt3_beta1, zdt3_beta2)
    run_bruteforce_verification()

    out_dir = Path("replication_outputs")
    write_outputs(
        [
            ("dense_front_beta_1", dense_result),
            ("zdt3_front_beta_1", zdt3_beta1),
            ("zdt3_front_beta_2", zdt3_beta2),
        ],
        out_dir,
    )
    print(f"\nWrote CSV and TikZ coordinate snippets to {out_dir}/")


if __name__ == "__main__":
    main()
