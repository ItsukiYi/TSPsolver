"""TSPLIB dataset loader with support for .tsp files.

Handles EUC_2D, GEO, ATT coordinate types. Optimal tours are validated against
known literature values since online .opt.tour files are unreliable to download.

Usage:
    from src.tsplib_loader import load_tsplib_instance, list_tsplib_instances
    points, opt_tour, opt_cost, name = load_tsplib_instance("eil51")
"""

import os
import gzip
import math
import re
from typing import List, Tuple, Optional, Dict

import numpy as np


# ============================================================
# Known optimal TSP tour lengths (from TSPLIB, widely cited)
# ============================================================
KNOWN_OPTIMAL_COSTS: Dict[str, float] = {
    "eil51": 426.0,
    "berlin52": 7542.0,
    "st70": 675.0,
    "eil76": 538.0,
    "pr76": 108159.0,
    "gr96": 55209.0,
    "rat99": 1211.0,
    "kroA100": 21282.0,
    "kroB100": 22141.0,
    "kroC100": 20749.0,
    "kroD100": 21294.0,
    "kroE100": 22068.0,
    "rd100": 7910.0,
    "eil101": 629.0,
    "lin105": 14379.0,
    "pr107": 44303.0,
    "pr124": 59030.0,
    "bier127": 118282.0,
    "ch130": 6110.0,
    "pr136": 96772.0,
    "gr137": 69853.0,
    "pr144": 58537.0,
    "ch150": 6528.0,
    "kroA150": 26524.0,
    "kroB150": 26130.0,
    "pr152": 73682.0,
    "u159": 42080.0,
    "si175": 21407.0,
    "brg180": 1950.0,
    "rat195": 2323.0,
    "d198": 15780.0,
    "kroA200": 29368.0,
    "kroB200": 29437.0,
    "gr202": 40160.0,
    "ts225": 126643.0,
    "tsp225": 3916.0,
    "pr226": 80369.0,
    "gr229": 134602.0,
    "gil262": 2378.0,
    "pr264": 49135.0,
    "a280": 2579.0,
    "pr299": 48191.0,
    "lin318": 42029.0,
    "linhp318": 41345.0,
    "rd400": 15281.0,
    "fl417": 11861.0,
    "gr431": 171414.0,
    "pr439": 107217.0,
    "pcb442": 50778.0,
    "d493": 35002.0,
    "att532": 27686.0,
    "ali535": 202339.0,
    "pa561": 2763.0,
    "u574": 36905.0,
    "rat575": 6773.0,
    "p654": 34643.0,
    "d657": 48912.0,
    "gr666": 294358.0,
    "u724": 41910.0,
    "rat783": 8806.0,
    "dsj1000": 18659688.0,
    "pr1002": 259045.0,
    "u1060": 224094.0,
    "vm1084": 239297.0,
    "pcb1173": 56892.0,
    "d1291": 50801.0,
    "rl1304": 252948.0,
    "rl1323": 270199.0,
    "nrw1379": 56638.0,
    "fl1400": 20127.0,
    "u1432": 152970.0,
    "fl1577": 22249.0,
    "d1655": 62128.0,
    "vm1748": 336556.0,
    "u1817": 57201.0,
    "rl1889": 316536.0,
    "d2103": 80450.0,
    "u2152": 64253.0,
    "u2319": 234256.0,
    "pr2392": 378032.0,
    "pcb3038": 137694.0,
    "fl3795": 28772.0,
    "fnl4461": 182566.0,
    "rl5915": 565530.0,
    "rl5934": 556045.0,
    "rl11849": 923288.0,
}


# ============================================================
# TSPLIB data directory
# ============================================================
_DEFAULT_TSPLIB_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "tsplib")


# ============================================================
# Coordinate parsing
# ============================================================

def _parse_euc_2d(points_dict: Dict[int, Tuple[float, float]]) -> np.ndarray:
    coords = [points_dict[i] for i in sorted(points_dict.keys())]
    return np.array(coords, dtype=np.float64)


def _parse_geo(points_dict: Dict[int, Tuple[float, float]]) -> np.ndarray:
    """GEO: Geographical coordinates (lat/lon) → 2D Euclidean."""
    coords = []
    for i in sorted(points_dict.keys()):
        lat_deg_val = points_dict[i][0]
        lon_deg_val = points_dict[i][1]

        lat_deg = int(lat_deg_val)
        lat_min = lat_deg_val - lat_deg
        lat_rad = math.pi * (lat_deg + 5.0 * lat_min / 3.0) / 180.0

        lon_deg = int(lon_deg_val)
        lon_min = lon_deg_val - lon_deg
        lon_rad = math.pi * (lon_deg + 5.0 * lon_min / 3.0) / 180.0

        R = 6378.388  # Earth radius in km
        q1 = math.cos(lon_rad)
        q2 = math.cos(lat_rad)
        q3 = math.cos(lat_rad)
        x = int(R * lon_rad * math.cos(lat_rad))
        y = int(R * lat_rad)

        coords.append([float(x), float(y)])

    return np.array(coords, dtype=np.float64)


def _parse_att(points_dict: Dict[int, Tuple[float, float]]) -> np.ndarray:
    coords = [points_dict[i] for i in sorted(points_dict.keys())]
    return np.array(coords, dtype=np.float64)


_COORD_PARSERS = {
    "EUC_2D": _parse_euc_2d,
    "GEO": _parse_geo,
    "ATT": _parse_att,
}


# ============================================================
# TSP file parser
# ============================================================

def parse_tsp_file(filepath: str) -> Tuple[np.ndarray, str, dict]:
    """Parse a .tsp or .tsp.gz file."""
    if filepath.endswith(".gz"):
        opener = gzip.open
        mode = "rt"
    else:
        opener = open
        mode = "r"

    with opener(filepath, mode) as f:
        content = f.read()

    name = "unknown"
    edge_weight_type = "EUC_2D"
    dimension = 0
    points_dict = {}
    in_coord_section = False

    for line in content.split("\n"):
        line = line.strip()
        if not line or line.startswith("COMMENT"):
            continue
        if line.startswith("NAME"):
            name = line.split(":")[-1].strip()
        elif line.startswith("EDGE_WEIGHT_TYPE"):
            edge_weight_type = line.split(":")[-1].strip()
        elif line.startswith("DIMENSION"):
            dimension = int(line.split(":")[-1].strip())
        elif line.startswith("NODE_COORD_SECTION"):
            in_coord_section = True
            continue
        elif line.startswith("EOF"):
            break
        if in_coord_section:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    node_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    points_dict[node_id] = (x, y)
                except (ValueError, IndexError):
                    continue

    parser = _COORD_PARSERS.get(edge_weight_type, _parse_euc_2d)
    points = parser(points_dict)

    metadata = {
        "name": name,
        "dimension": dimension,
        "edge_weight_type": edge_weight_type,
    }
    return points, name, metadata


def parse_tour_file(filepath: str) -> Optional[List[int]]:
    """Parse a .opt.tour file."""
    if not os.path.exists(filepath):
        return None
    if filepath.endswith(".gz"):
        opener = gzip.open
        mode = "rt"
    else:
        opener = open
        mode = "r"
    with opener(filepath, mode) as f:
        content = f.read()
    tour = []
    in_tour_section = False
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("TOUR_SECTION"):
            in_tour_section = True
            continue
        if line.startswith("-1") or line.startswith("EOF"):
            break
        if in_tour_section:
            try:
                node_id = int(line)
                if node_id > 0 and node_id not in tour:
                    tour.append(node_id - 1)
            except ValueError:
                continue
    if tour and tour[0] != tour[-1]:
        tour.append(tour[0])
    return tour if len(tour) > 0 else None


# ============================================================
# Main API
# ============================================================

def load_tsplib_instance(
    name: str,
    data_dir: str = _DEFAULT_TSPLIB_DIR,
) -> Tuple[np.ndarray, Optional[List[int]], Optional[float], str]:
    """Load a TSPLIB instance.

    Returns:
        (points, optimal_tour, known_optimal_cost, display_name)
        If no .opt.tour file exists but known optimal cost is available,
        optimal_tour will be None but known_optimal_cost will be populated.
    """
    tsp_path = os.path.join(data_dir, f"{name}.tsp.gz")
    if not os.path.exists(tsp_path):
        tsp_path = os.path.join(data_dir, f"{name}.tsp")
    if not os.path.exists(tsp_path):
        raise FileNotFoundError(f"TSP file not found: {name}.tsp[.gz]")

    points, instance_name, metadata = parse_tsp_file(tsp_path)

    # Try to load optimal tour file
    opt_tour = None
    for suffix in [".opt.tour.gz", ".opt.tour", ".tour.gz", ".tour"]:
        opt_path = os.path.join(data_dir, f"{name}{suffix}")
        if os.path.exists(opt_path):
            opt_tour = parse_tour_file(opt_path)
            if opt_tour is not None:
                break

    # Known optimal cost
    opt_cost = KNOWN_OPTIMAL_COSTS.get(name)

    display_name = f"{instance_name} (n={len(points)})"
    return points, opt_tour, opt_cost, display_name


def list_tsplib_instances(data_dir: str = _DEFAULT_TSPLIB_DIR) -> List[str]:
    """List all available TSPLIB instances."""
    instances = set()
    if not os.path.isdir(data_dir):
        return []
    for fname in os.listdir(data_dir):
        match = re.match(r"(.+)\.tsp(?:\.gz)?$", fname)
        if match:
            instances.add(match.group(1))
    return sorted(instances)


def load_all_tsplib(
    data_dir: str = _DEFAULT_TSPLIB_DIR,
) -> Dict[str, Tuple[np.ndarray, Optional[List[int]], Optional[float], str]]:
    """Load all available TSPLIB instances.

    Returns:
        dict mapping instance name → (points, opt_tour, opt_cost, display_name)
    """
    instances = {}
    for name in list_tsplib_instances(data_dir):
        try:
            instances[name] = load_tsplib_instance(name, data_dir)
        except Exception as e:
            print(f"Warning: could not load {name}: {e}")
    return instances


# ============================================================
# Generate optimal tours using our solver
# ============================================================

def generate_optimal_tours(
    data_dir: str = _DEFAULT_TSPLIB_DIR,
    max_2opt_iter: int = 10000,
    tolerance: float = 0.001,
) -> Dict[str, List[int]]:
    """Generate near-optimal tours for TSPLIB instances using Christofides+2opt.

    Validates against known optimal costs and warns if gap > tolerance.

    Returns:
        dict mapping instance name → 0-indexed tour
    """
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.algorithms import christofides_with_2opt
    from src.utils import compute_distance_matrix, tour_cost

    tours = {}
    for name in list_tsplib_instances(data_dir):
        points, _, known_opt, display = load_tsplib_instance(name, data_dir)
        print(f"Solving {name} (n={len(points)})...", end=" ", flush=True)

        best_tour = None
        best_cost = float("inf")
        # Run multiple times with different seeds for best result
        for seed in range(5):
            np.random.seed(seed)
            tour, _ = christofides_with_2opt(
                points, max_2opt_iterations=max_2opt_iter
            )
            cost = tour_cost(compute_distance_matrix(points), tour)
            if cost < best_cost:
                best_cost = cost
                best_tour = tour

        tours[name] = best_tour

        if known_opt is not None:
            gap = (best_cost / known_opt - 1) * 100
            status = "OK" if gap < tolerance * 100 else f"GAP={gap:.2f}%"
            print(f"cost={best_cost:.2f} (opt={known_opt}, {status})")
        else:
            print(f"cost={best_cost:.2f}")

    return tours


# ============================================================
# Export to DIFUSCO format
# ============================================================

def export_tsplib_to_difusco(
    output_file: str,
    data_dir: str = _DEFAULT_TSPLIB_DIR,
    generate_tours: bool = True,
):
    """Export TSPLIB instances with optimal/near-optimal tours to DIFUSCO format.

    Args:
        output_file: path for the output data file
        data_dir: TSPLIB directory
        generate_tours: if True, generate tours with our solver;
                        if False, only export instances with existing .opt.tour files
    """
    from src.utils import compute_distance_matrix, tour_cost

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Clear file
    with open(output_file, "w") as f:
        pass

    if generate_tours:
        tours = generate_optimal_tours(data_dir)
    else:
        tours = {}

    count = 0
    for name in list_tsplib_instances(data_dir):
        points, opt_tour, opt_cost, display = load_tsplib_instance(name, data_dir)

        # Use existing optimal tour if available, otherwise generated tour
        if opt_tour is not None:
            tour = opt_tour
        elif name in tours:
            tour = tours[name]
        else:
            continue

        # Write in DIFUSCO format (1-indexed)
        coord_str = " ".join(f"{x} {y}" for x, y in points)
        tour_1idx = [str(t + 1) for t in tour]
        tour_str = " ".join(tour_1idx)

        with open(output_file, "a") as f:
            f.write(f"{coord_str} output {tour_str}\n")
        count += 1

    print(f"Exported {count} instances to {output_file}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TSPLIB loader and converter")
    parser.add_argument("--list", action="store_true", help="List available instances")
    parser.add_argument("--load", type=str, help="Load specific instance")
    parser.add_argument("--generate-tours", action="store_true", help="Generate optimal tours with our solver")
    parser.add_argument("--export", type=str, help="Export all instances to DIFUSCO format file")
    parser.add_argument("--data-dir", type=str, default=_DEFAULT_TSPLIB_DIR)

    args = parser.parse_args()

    if args.list:
        names = list_tsplib_instances(args.data_dir)
        print(f"Available TSPLIB instances ({len(names)}):")
        for name in names:
            pts, opt_tour, opt_cost, display = load_tsplib_instance(name, args.data_dir)
            has_tour = "Y" if opt_tour is not None else "N"
            has_opt = f"opt={opt_cost}" if opt_cost else "opt=?"
            print(f"  {name:15s}  n={len(pts):5d}  tour={has_tour}  {has_opt}")

    elif args.load:
        pts, opt_tour, opt_cost, display = load_tsplib_instance(args.load, args.data_dir)
        print(f"Loaded: {display}")
        print(f"Points: {pts.shape}")
        from src.utils import compute_distance_matrix, tour_cost
        if opt_tour:
            cost = tour_cost(compute_distance_matrix(pts), opt_tour)
            print(f"Optimal tour cost: {cost:.4f}")
        if opt_cost:
            print(f"Known optimal: {opt_cost}")

    elif args.generate_tours:
        generate_optimal_tours(args.data_dir)

    elif args.export:
        export_tsplib_to_difusco(args.export, args.data_dir, generate_tours=True)

    else:
        parser.print_help()
