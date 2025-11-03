import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy.typing as npt
import json
import argparse
import heapq
import time
from mccormick_relaxation import node_lower_bound, node_upper_bound

@dataclass(order=True)
class _PQItem:
    # priority queue ordered by node lower bound (best-first / best-bound)
    lb_value: float
    count: int
    node: "BBNode"

@dataclass
class BBNode:
    lb: np.ndarray
    ub: np.ndarray
    depth: int

@dataclass
class BnBResult:
    success: bool
    x: Optional[np.ndarray]
    obj: Optional[float]
    global_lb: float
    nodes_expanded: int
    nodes_pruned: int
    nodes_generated: int
    time_sec: float

def branch_and_bound_qcqp(
    n: int,
    m: int,
    c: List[float],
    H: npt.NDArray[np.float64],
    Q: npt.NDArray[np.float64],
    A: npt.NDArray[np.float64],
    b: List[float],
    lb: List[float],
    ub: List[float],
    *,
    # B&B controls
    abs_gap_tol: float = 1e-4,
    rel_gap_tol: float = 1e-3,
    var_width_tol: float = 1e-6,
    max_nodes: int = 2000,
    time_limit_sec: Optional[float] = None,
    # Solver controls
    lb_maxiter: int = 1000,
    lb_provide_analytic_jacobians: bool = True,
    lb_sym_tol: float = 1e-8,
    ub_extra_starts: int = 10,
    ub_rng_seed: Optional[int] = 42,
    ub_tol_eq: float = 1e-6,
    ub_maxiter: int = 2000,
    verbose: bool = False,
) -> BnBResult:
    """
    A minimal best-bound Branch & Bound using:
      - node_lower_bound(...) for relaxation (McCormick)  -> lower bounds
      - node_upper_bound(...) for feasible search         -> upper bounds

    Notes:
      • Only bilinear terms are allowed in Q (your _assert_shapes already enforces zero diagonal).
      • If H is indefinite, the LB is a *candidate* bound (SLSQP local optimum).
        This is still practical but not globally certified; keep that in mind when interpreting gaps.
    """
    t0 = time.time()

    # Initialize global best UB and LB trackers
    best_x: Optional[np.ndarray] = None
    best_obj: Optional[float] = None
    global_lb: float = -np.inf

    # Priority queue of nodes (best lower bound first)
    pq: List[_PQItem] = []
    counter = 0

    # Helper: evaluate (LB, LB-res) at a node
    def _solve_lb(curr_lb: np.ndarray, curr_ub: np.ndarray):
        try:
            lb_res = node_lower_bound(
                n, m, c, H, Q, A, b, curr_lb, curr_ub,
                provide_analytic_jacobians=lb_provide_analytic_jacobians,
                maxiter=lb_maxiter, verbose=False, sym_tol=lb_sym_tol
            )
            # SLSQP returns a result object even when success=False; treat those as infeasible/invalid bounds.
            if not lb_res.res.success or not np.isfinite(lb_res.res.fun):
                return None, np.inf
            return lb_res, float(lb_res.res.fun)
        except Exception:
            # Inconsistent box (e.g., McCormick needs finite bounds) -> infeasible
            return None, np.inf

    # Helper: try to get a UB from this node (warm-started by LB x)
    def _solve_ub(curr_lb: np.ndarray, curr_ub: np.ndarray, x0: Optional[np.ndarray]):
        try:
            ub_res = node_upper_bound(
                n, m, c, H, Q, A, b, curr_lb, curr_ub,
                x0=x0, extra_starts=ub_extra_starts, rng_seed=ub_rng_seed,
                tol_eq=ub_tol_eq, maxiter=ub_maxiter, verbose=False
            )
            return ub_res
        except Exception:
            return None

    # Choose branching variable: widest interval among variables that appear in bilinear terms,
    # falling back to globally widest if none qualify.
    def _pick_branch_var(curr_lb: np.ndarray, curr_ub: np.ndarray, bilinear_terms: List[Tuple[int, int]]) -> Optional[int]:
        widths = curr_ub - curr_lb
        candidate_mask = np.zeros(n, dtype=bool)
        for i, j in bilinear_terms:
            candidate_mask[i] = True
            candidate_mask[j] = True
        # Prefer variables in bilinear pairs
        if np.any(candidate_mask):
            idx = np.argmax(np.where(candidate_mask, widths, -1.0))
            if widths[idx] > var_width_tol:
                return int(idx)
        # Otherwise, pick globally widest
        idx = int(np.argmax(widths))
        if widths[idx] > var_width_tol:
            return idx
        return None

    # Seed root node
    root = BBNode(lb=np.asarray(lb, dtype=float), ub=np.asarray(ub, dtype=float), depth=0)
    root_lb_res, root_lb_val = _solve_lb(root.lb, root.ub)
    nodes_generated = 1
    nodes_expanded = 0
    nodes_pruned = 0

    if root_lb_res is None or not np.isfinite(root_lb_val):
        # No feasible relaxation at root
        return BnBResult(False, None, None, global_lb=-np.inf, nodes_expanded=0,
                         nodes_pruned=1, nodes_generated=1, time_sec=time.time() - t0)

    # Try an initial UB from the root
    ub_res0 = _solve_ub(root.lb, root.ub, x0=root_lb_res.x if root_lb_res is not None else None)
    if ub_res0 and ub_res0.success:
        best_x = ub_res0.x
        best_obj = float(ub_res0.obj)

    heapq.heappush(pq, _PQItem(lb_value=root_lb_val, count=counter, node=root))
    counter += 1
    global_lb = min(global_lb, root_lb_val) if np.isfinite(global_lb) else root_lb_val

    # Main loop
    while pq and nodes_expanded < max_nodes:
        if time_limit_sec is not None and (time.time() - t0) >= time_limit_sec:
            break

        # Check global gap for early termination
        if best_obj is not None:
            incumbent = best_obj
            frontier_best_lb = pq[0].lb_value
            abs_gap = incumbent - frontier_best_lb
            rel_gap = abs_gap / (1.0 + abs(abs(incumbent)))
            if abs_gap <= abs_gap_tol or rel_gap <= rel_gap_tol:
                break

        # Pop best-bound node
        item = heapq.heappop(pq)
        node = item.node
        node_lb_val = item.lb_value
        global_lb = min(global_lb, node_lb_val) if np.isfinite(global_lb) else node_lb_val

        # Prune by bound
        if (best_obj is not None) and (node_lb_val >= best_obj - abs_gap_tol):
            nodes_pruned += 1
            continue

        # Resolve LB at node (fresh solve to get bilinear terms / x for branching & UB warm-start)
        lb_res, lb_val = _solve_lb(node.lb, node.ub)
        nodes_expanded += 1
        if lb_res is None or not np.isfinite(lb_val):
            nodes_pruned += 1
            continue

        # Try UB at this node
        ub_res = _solve_ub(node.lb, node.ub, x0=lb_res.x)
        if ub_res and ub_res.success:
            obj_val = float(ub_res.obj)
            if (best_obj is None) or (obj_val < best_obj):
                best_obj = obj_val
                best_x = ub_res.x

        # Prune again w.r.t. updated incumbent
        if (best_obj is not None) and (lb_val >= best_obj - abs_gap_tol):
            nodes_pruned += 1
            continue

        # Choose branching variable
        i = _pick_branch_var(node.lb, node.ub, lb_res.bilinear_terms)
        if i is None:
            # Box is effectively integral/tight; nothing to branch on
            nodes_pruned += 1
            continue

        # Midpoint split
        mid = 0.5 * (node.lb[i] + node.ub[i])

        # Left child: ub_i = mid
        lb_L = node.lb.copy(); ub_L = node.ub.copy(); ub_L[i] = mid
        # Right child: lb_i = mid
        lb_R = node.lb.copy(); ub_R = node.ub.copy(); lb_R[i] = mid

        # Evaluate LBs for children once (cheap screening) to order by bound
        for child_lb, child_ub in ((lb_L, ub_L), (lb_R, ub_R)):
            child_res, child_val = _solve_lb(child_lb, child_ub)
            if child_res is None or not np.isfinite(child_val):
                nodes_pruned += 1
                continue
            # Prune if dominated by incumbent
            if (best_obj is not None) and (child_val >= best_obj - abs_gap_tol):
                nodes_pruned += 1
                continue
            child_node = BBNode(lb=child_lb, ub=child_ub, depth=node.depth + 1)
            heapq.heappush(pq, _PQItem(lb_value=child_val, count=counter, node=child_node))
            counter += 1
            nodes_generated += 1

    t1 = time.time()
    # Final global LB is the best bound among remaining active nodes (or what we tracked)
    if pq:
        global_lb = min(global_lb, pq[0].lb_value) if np.isfinite(global_lb) else pq[0].lb_value

    return BnBResult(
        success=(best_x is not None),
        x=best_x,
        obj=(float(best_obj) if best_obj is not None else None),
        global_lb=float(global_lb) if np.isfinite(global_lb) else -np.inf,
        nodes_expanded=nodes_expanded,
        nodes_pruned=nodes_pruned,
        nodes_generated=nodes_generated,
        time_sec=t1 - t0,
    )
# =========================
# CLI helpers
# =========================

def _load_problem_from_json(path: str):
    """
    JSON schema:
    {
      "n": 3,
      "m": 2,
      "c": [1, 1, 0],
      "b": [8, 15],
      "A": [[0,0,1],[0,1.8,0]],           // shape (m, n)
      "H": [[0,0,0],[0,0,0],[0,0,1]],      // shape (n, n)
      "Q": [                               // list of m matrices (n x n) or a 3D array
        [[0,0.5,0],[0.5,0,0.5],[0,0.5,0]],
        [[0,0.5,0],[0.5,0,0],[0,0,0]]
      ],
      "lb": [0, 0, 0],
      "ub": [10, 10, 10]
    }

    Each Q_k must be symmetric with zero diagonal (bilinear-only). Nonzero diagonals raise an error.
    """
    with open(path, "r") as f:
        data = json.load(f)
    n = int(data["n"])
    m = int(data["m"])
    c = np.asarray(data["c"], dtype=float)
    b = np.asarray(data["b"], dtype=float)
    A = np.asarray(data["A"], dtype=float)
    H = np.asarray(data["H"], dtype=float)
    Q_raw = np.asarray(data["Q"], dtype=float)
    if Q_raw.ndim == 3 and Q_raw.shape == (m, n, n):
        Q = np.transpose(Q_raw, (1, 2, 0))  # if provided as (m, n, n), convert to (n, n, m)
    elif Q_raw.ndim == 3 and Q_raw.shape == (n, n, m):
        Q = Q_raw
    elif Q_raw.ndim == 2 and Q_raw.shape == (m, n*n):
        # allow flattened Qs: each row is vec(Q_k) in row-major
        Q = Q_raw.reshape(m, n, n).transpose(1, 2, 0)
    else:
        # try list-of-matrices
        Q_list = [np.asarray(M, dtype=float) for M in Q_raw]
        if len(Q_list) != m:
            raise ValueError("Q must contain m matrices")
        for M in Q_list:
            if M.shape != (n, n):
                raise ValueError("Each Q_k must be (n, n)")
        Q = np.dstack(Q_list)  # (n, n, m)
    lb = np.asarray(data["lb"], dtype=float)
    ub = np.asarray(data["ub"], dtype=float)
    return n, m, c, H, Q, A, b, lb, ub

def _build_arg_parser():
    p = argparse.ArgumentParser(description="bilinear-equality QCQP node lower/upper bounds + pipeline")
    p.add_argument("--config", type=str, required=True,
                   help="Path to JSON file with problem data (see schema in _load_problem_from_json docstring).")
    # What to run
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--pipeline", action="store_true", help="Run LB->UB pipeline (default if neither LB/UB-only set).")
    mode.add_argument("--lb-only", action="store_true", help="Run only the lower-bound relaxation.")
    mode.add_argument("--ub-only", action="store_true", help="Run only the upper-bound bilinear-equality QCQP.")
    # LB options
    p.add_argument("--lb-verbose", action="store_true")
    p.add_argument("--lb-maxiter", type=int, default=2000)
    p.add_argument("--lb-no-analytic-jacobians", action="store_true", help="Disable analytic Jacobians for LB.")
    p.add_argument("--lb-sym-tol", type=float, default=1e-8)
    # UB options
    p.add_argument("--ub-extra-starts", type=int, default=20)
    p.add_argument("--ub-rng-seed", type=int, default=42)
    p.add_argument("--ub-tol-eq", type=float, default=1e-6)
    p.add_argument("--ub-maxiter", type=int, default=2000)
    p.add_argument("--ub-verbose", action="store_true")
    return p
# -----------------------------------------------------------------
# __main__
# -----------------------------------------------------------------

if __name__ == "__main__":
    # (Re-use your JSON loader)
    args = _build_arg_parser().parse_args()
    n, m, c, H, Q, A, b, lb, ub = _load_problem_from_json(args.config)

    res = branch_and_bound_qcqp(
        n, m, c, H, Q, A, b, lb, ub,
        abs_gap_tol=1e-4, rel_gap_tol=1e-3,
        max_nodes=1000, time_limit_sec=30.0,
        ub_extra_starts=10, ub_tol_eq=1e-6,
        verbose=True
    )

    print("\n=== Branch & Bound ===")
    print(f"Success: {res.success}")
    print(f"Best UB: {res.obj}")
    print(f"Global LB: {res.global_lb}")
    if res.obj is not None:
        print(f"Abs gap: {res.obj - res.global_lb}")
        print(f"Rel gap: {(res.obj - res.global_lb)/(1+abs(res.obj))}")
    print(f"Nodes expanded: {res.nodes_expanded}, generated: {res.nodes_generated}, pruned: {res.nodes_pruned}")
    if res.x is not None:
        print("x*:", res.x)