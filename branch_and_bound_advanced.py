# =========================
# Advanced Branch & Bound (uses node_lower_bound / node_upper_bound)
# =========================
import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional
import numpy.typing as npt
import json
import argparse
import heapq
import time
from mccormick_relaxation import node_lower_bound, node_upper_bound

# ---------- Small utilities ----------

@dataclass(order=True)
class _PQItem:
    lb_value: float
    counter: int
    node: "BBNode"

@dataclass
class BBNode:
    lb: np.ndarray
    ub: np.ndarray
    depth: int
    bound: float = np.inf   # store LB to help reordering / logging

@dataclass
class BnBStats:
    nodes_generated: int = 0
    nodes_expanded: int = 0
    nodes_pruned: int = 0
    lb_solves: int = 0
    ub_solves: int = 0
    time_sec: float = 0.0

@dataclass
class BnBResult:
    success: bool
    x: Optional[np.ndarray]
    obj: Optional[float]
    global_lb: float
    stats: BnBStats
    message: str

@dataclass
class BnBOptions:
    # Gaps & limits
    abs_gap_tol: float = 1e-4
    rel_gap_tol: float = 1e-3
    var_width_tol: float = 1e-7
    max_nodes: int = 5000
    time_limit_sec: Optional[float] = None
    log_every_sec: float = 5.0

    # LB solver
    lb_maxiter: int = 4000
    lb_provide_analytic_jacobians: bool = True
    lb_sym_tol: float = 1e-8

    # UB solver
    ub_extra_starts: int = 40
    ub_rng_seed: Optional[int] = 42
    ub_tol_eq: float = 1e-6
    ub_maxiter: int = 4000

    # Branching
    strong_branch_topk: int = 3        # strong branch among top-K candidates
    prepartition_var: Optional[int] = None  # index to pre-partition at root (None = auto pick)
    prepartition_pieces: int = 3

    # Heuristics
    enable_implied_bounds: bool = True
    enable_warmstart_from_lb: bool = True


# ---------- Implied-bound tightening on singleton rows (cheap OBBT) ----------

def _implied_bound_tighten_singletons(
    n: int,
    Q: npt.NDArray[np.float64],
    A: npt.NDArray[np.float64],
    b: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    *,
    sym_tol: float = 1e-8,
    max_passes: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tighten variable bounds using rows that have exactly one linear variable in A:
        a * x_v + sum_{i<j} (2*Q[i,j,k]) * (x_i x_j) = b_k  (k-th row)
    Over current box [lb,ub], bound each product by corners and propagate to x_v.
    """
    lb = lb.copy()
    ub = ub.copy()
    m = Q.shape[2]

    # Collect unique bilinear pairs and alphas per row
    pairs: List[Tuple[int,int]] = []
    pair_idx: Dict[Tuple[int,int], int] = {}
    for k in range(m):
        Qi = Q[:, :, k]
        for i in range(n):
            for j in range(i+1, n):
                if abs(Qi[i, j]) > sym_tol:
                    if (i, j) not in pair_idx:
                        pair_idx[(i, j)] = len(pairs)
                        pairs.append((i, j))
    P = len(pairs)
    alphas = np.zeros((m, P))
    for k in range(m):
        Qi = Q[:, :, k]
        for p, (i, j) in enumerate(pairs):
            if abs(Qi[i, j]) > sym_tol:
                alphas[k, p] = 2.0 * Qi[i, j]

    def prod_bounds(i, j):
        vals = [lb[i]*lb[j], lb[i]*ub[j], ub[i]*lb[j], ub[i]*ub[j]]
        return min(vals), max(vals)

    for _ in range(max_passes):
        changed = False
        for k in range(m):
            nz = np.nonzero(np.abs(A[k, :]) > 0)[0]
            if nz.size != 1:
                continue
            v = int(nz[0])
            a = A[k, v]
            if abs(a) < 1e-15:
                continue

            # interval for sum alpha*y
            y_lo, y_hi = 0.0, 0.0
            for p, (i, j) in enumerate(pairs):
                alpha = alphas[k, p]
                if abs(alpha) < 1e-15:
                    continue
                lo, hi = prod_bounds(i, j)
                if alpha >= 0:
                    y_lo += alpha * lo
                    y_hi += alpha * hi
                else:
                    y_lo += alpha * hi
                    y_hi += alpha * lo

            # a x_v = b_k - S_y  =>  x_v in [(b - y_hi)/a, (b - y_lo)/a] reordered by sign(a)
            lo = (b[k] - y_hi) / a
            hi = (b[k] - y_lo) / a
            if lo > hi:
                lo, hi = hi, lo

            new_lb = max(lb[v], lo)
            new_ub = min(ub[v], hi)
            if new_lb > new_ub + 1e-12:
                # Infeasible — leave as-is; node will be declared infeasible by LB solve
                continue
            if new_lb > lb[v] + 1e-12:
                lb[v] = new_lb; changed = True
            if new_ub < ub[v] - 1e-12:
                ub[v] = new_ub; changed = True
        if not changed:
            break
    return lb, ub


# ---------- Branching rules ----------

def _degrees_from_pairs(n: int, pairs: List[Tuple[int,int]]) -> np.ndarray:
    deg = np.zeros(n, dtype=int)
    for i, j in pairs:
        deg[i] += 1; deg[j] += 1
    return deg

def _pick_candidates_by_score(lb: np.ndarray, ub: np.ndarray, pairs: List[Tuple[int,int]], var_width_tol: float, topk: int) -> List[int]:
    widths = ub - lb
    if len(pairs) == 0:
        idx = int(np.argmax(widths))
        return [idx] if widths[idx] > var_width_tol else []
    deg = _degrees_from_pairs(len(lb), pairs)
    score = (deg.astype(float) + 1e-12) * widths  # degree × width
    order = np.argsort(-score)
    cands = [int(i) for i in order if widths[i] > var_width_tol][:topk]
    return cands

def _midpoint_split(lb: np.ndarray, ub: np.ndarray, i: int) -> Tuple[Tuple[np.ndarray,np.ndarray], Tuple[np.ndarray,np.ndarray]]:
    mid = 0.5 * (lb[i] + ub[i])
    L_lb, L_ub = lb.copy(), ub.copy(); L_ub[i] = mid
    R_lb, R_ub = lb.copy(), ub.copy(); R_lb[i] = mid
    return (L_lb, L_ub), (R_lb, R_ub)


# ---------- Core solver ----------

def branch_and_bound_advanced(
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
    opts: Optional[BnBOptions] = None,
    logger: Optional[Callable[[str], None]] = None
) -> BnBResult:
    """
    Advanced B&B for bilinear-equality QCQPs using node_lower_bound / node_upper_bound.
    """
    if opts is None:
        opts = BnBOptions()
    if logger is None:
        logger = lambda s: None

    t0 = time.time()
    stats = BnBStats()
    c = np.asarray(c, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    lb0 = np.asarray(lb, dtype=float).reshape(-1)
    ub0 = np.asarray(ub, dtype=float).reshape(-1)
    H = np.asarray(H, dtype=float)
    Q = np.asarray(Q, dtype=float)
    A = np.asarray(A, dtype=float)

    best_x: Optional[np.ndarray] = None
    best_obj: Optional[float] = None
    global_lb = np.inf  # best lower bound across active nodes is min of PQ keys

    # LB/UB wrappers (count solves; allow quick LB passes)
    def solve_lb(box_lb: np.ndarray, box_ub: np.ndarray, *, maxiter: Optional[int] = None):
        stats.lb_solves += 1
        res = node_lower_bound(
            n, m, c, H, Q, A, b, box_lb, box_ub,
            provide_analytic_jacobians=opts.lb_provide_analytic_jacobians,
            maxiter=(opts.lb_maxiter if maxiter is None else maxiter),
            verbose=False, sym_tol=opts.lb_sym_tol
        )
        if not np.isfinite(res.res.fun):
            return None, np.inf, None
        return res, float(res.res.fun), res.x

    def solve_ub(box_lb: np.ndarray, box_ub: np.ndarray, x0: Optional[np.ndarray]):
        stats.ub_solves += 1
        return node_upper_bound(
            n, m, c, H, Q, A, b, box_lb, box_ub,
            x0=x0 if opts.enable_warmstart_from_lb else None,
            extra_starts=opts.ub_extra_starts,
            rng_seed=opts.ub_rng_seed,
            tol_eq=opts.ub_tol_eq,
            maxiter=opts.ub_maxiter,
            verbose=False
        )

    # Root node + optional implied-bound tightening
    root_lb, root_ub = lb0.copy(), ub0.copy()
    if opts.enable_implied_bounds:
        root_lb, root_ub = _implied_bound_tighten_singletons(n, Q, A, b, root_lb, root_ub,
                                                             sym_tol=opts.lb_sym_tol)

    # Root pre-partitioning (static piecewise McCormick)
    def pick_root_var():
        try:
            # A quick LB to extract bilinear pairs and widths
            lbres, val, _ = solve_lb(root_lb, root_ub, maxiter=min(400, opts.lb_maxiter))
            if lbres is None:
                return None
            widths = root_ub - root_lb
            if len(lbres.bilinear_terms) == 0:
                return int(np.argmax(widths))
            deg = _degrees_from_pairs(n, lbres.bilinear_terms)
            score = (deg.astype(float) + 1e-12) * widths
            return int(np.argmax(score))
        except Exception:
            return None

    initial_nodes: List[BBNode] = []
    if opts.prepartition_pieces >= 2:
        pv = opts.prepartition_var if opts.prepartition_var is not None else pick_root_var()
        if pv is not None and root_ub[pv] - root_lb[pv] > opts.var_width_tol:
            cuts = np.linspace(root_lb[pv], root_ub[pv], opts.prepartition_pieces + 1)
            for k in range(opts.prepartition_pieces):
                lb_k = root_lb.copy(); ub_k = root_ub.copy()
                lb_k[pv], ub_k[pv] = cuts[k], cuts[k+1]
                if opts.enable_implied_bounds:
                    lb_k, ub_k = _implied_bound_tighten_singletons(n, Q, A, b, lb_k, ub_k,
                                                                   sym_tol=opts.lb_sym_tol)
                initial_nodes.append(BBNode(lb=lb_k, ub=ub_k, depth=0))
        else:
            initial_nodes.append(BBNode(lb=root_lb, ub=root_ub, depth=0))
    else:
        initial_nodes.append(BBNode(lb=root_lb, ub=root_ub, depth=0))

    # Build PQ (evaluate LBs of initial nodes)
    pq: List[_PQItem] = []
    counter = 0
    last_log = time.time()

    for nd in initial_nodes:
        try:
            lbres, val, xwarm = solve_lb(nd.lb, nd.ub)
        except Exception:
            lbres, val, xwarm = None, np.inf, None
        if not np.isfinite(val) or lbres is None:
            stats.nodes_pruned += 1
            continue
        nd.bound = val
        heapq.heappush(pq, _PQItem(lb_value=val, counter=counter, node=nd))
        counter += 1
        stats.nodes_generated += 1
        # Initial UB try
        ubres = solve_ub(nd.lb, nd.ub, xwarm)
        if ubres and ubres.success:
            if (best_obj is None) or (ubres.obj < best_obj):
                best_obj = float(ubres.obj); best_x = ubres.x

    # Main loop
    while pq and stats.nodes_expanded < opts.max_nodes:
        now = time.time()
        if opts.time_limit_sec is not None and (now - t0) >= opts.time_limit_sec:
            break

        # Global LB is current best PQ head
        frontier_lb = pq[0].lb_value
        global_lb = frontier_lb

        # Gap stopping test
        if best_obj is not None:
            abs_gap = best_obj - global_lb
            rel_gap = abs_gap / (1.0 + abs(best_obj))
            if abs_gap <= opts.abs_gap_tol or rel_gap <= opts.rel_gap_tol:
                break

        # Logging
        if opts.log_every_sec and (now - last_log) >= opts.log_every_sec:
            logger(f"[B&B] time={now - t0:.1f}s  LB={global_lb:.6g}  "
                   f"UB={best_obj:.6g}  gap={abs(best_obj - global_lb):.3g}  "
                   f"nodes: gen={stats.nodes_generated} exp={stats.nodes_expanded} pruned={stats.nodes_pruned}")
            last_log = now

        # Pop best-bound node
        item = heapq.heappop(pq)
        node = item.node
        stats.nodes_expanded += 1

        # (Optional) refresh tightening at node
        if opts.enable_implied_bounds:
            node.lb, node.ub = _implied_bound_tighten_singletons(n, Q, A, b, node.lb, node.ub,
                                                                 sym_tol=opts.lb_sym_tol)

        # Recompute LB (to get fresh bilinear terms + warm start)
        lbres, node_lb_val, xwarm = solve_lb(node.lb, node.ub)
        if (lbres is None) or (not np.isfinite(node_lb_val)):
            stats.nodes_pruned += 1
            continue

        # Bound pruning
        if (best_obj is not None) and (node_lb_val >= best_obj - opts.abs_gap_tol):
            stats.nodes_pruned += 1
            continue

        # UB attempt
        ubres = solve_ub(node.lb, node.ub, xwarm)
        if ubres and ubres.success:
            if (best_obj is None) or (ubres.obj < best_obj):
                best_obj = float(ubres.obj); best_x = ubres.x

        # Re-check pruning after new incumbent
        if (best_obj is not None) and (node_lb_val >= best_obj - opts.abs_gap_tol):
            stats.nodes_pruned += 1
            continue

        # Choose branching variable: degree×width, then light strong branching among top-K
        pairs = lbres.bilinear_terms
        cand_idx = _pick_candidates_by_score(node.lb, node.ub, pairs, opts.var_width_tol, opts.strong_branch_topk)
        if not cand_idx:
            # Nothing wide enough to branch on (box tight) -> prune
            stats.nodes_pruned += 1
            continue

        # Strong-branch probe: evaluate child LBs quickly and pick best variable
        best_i = None
        best_gain = -np.inf
        for i in cand_idx:
            (L_lb, L_ub), (R_lb, R_ub) = _midpoint_split(node.lb, node.ub, i)
            # Quick implied bounding on children to improve probe quality
            if opts.enable_implied_bounds:
                L_lb, L_ub = _implied_bound_tighten_singletons(n, Q, A, b, L_lb, L_ub, sym_tol=opts.lb_sym_tol, max_passes=2)
                R_lb, R_ub = _implied_bound_tighten_singletons(n, Q, A, b, R_lb, R_ub, sym_tol=opts.lb_sym_tol, max_passes=2)
            # Fast LB (reduced iterations)
            L_res, L_val, _ = solve_lb(L_lb, L_ub, maxiter=min(400, opts.lb_maxiter))
            R_res, R_val, _ = solve_lb(R_lb, R_ub, maxiter=min(400, opts.lb_maxiter))
            child_best = min(L_val, R_val)
            # choose variable that maximizes the *worse* child (i.e., raises the bound)
            child_worst = max(L_val, R_val)
            # “gain” relative to current node’s LB
            gain = min(child_worst, child_best) - node_lb_val
            if gain > best_gain:
                best_gain = gain
                best_i = i

        # Split on chosen variable
        i = best_i if best_i is not None else cand_idx[0]
        (L_lb, L_ub), (R_lb, R_ub) = _midpoint_split(node.lb, node.ub, i)
        # Optional tightening
        if opts.enable_implied_bounds:
            L_lb, L_ub = _implied_bound_tighten_singletons(n, Q, A, b, L_lb, L_ub, sym_tol=opts.lb_sym_tol)
            R_lb, R_ub = _implied_bound_tighten_singletons(n, Q, A, b, R_lb, R_ub, sym_tol=opts.lb_sym_tol)

        # Child LB screening (with standard LB effort)
        children = []
        for c_lb, c_ub in ((L_lb, L_ub), (R_lb, R_ub)):
            try:
                c_lbres, c_val, c_x = solve_lb(c_lb, c_ub)
            except Exception:
                c_lbres, c_val, c_x = None, np.inf, None
            if (c_lbres is None) or (not np.isfinite(c_val)):
                stats.nodes_pruned += 1
                continue
            # Incumbent pruning
            if (best_obj is not None) and (c_val >= best_obj - opts.abs_gap_tol):
                stats.nodes_pruned += 1
                continue
            nd = BBNode(lb=c_lb, ub=c_ub, depth=node.depth + 1, bound=c_val)
            heapq.heappush(pq, _PQItem(lb_value=c_val, counter=counter, node=nd))
            counter += 1
            stats.nodes_generated += 1

    # Wrap up
    stats.time_sec = time.time() - t0
    if best_obj is None:
        # No feasible solution found
        msg = "No feasible solution found within limits." if pq == [] else "Stopped by limits; no feasible incumbent."
        return BnBResult(False, None, None, global_lb=(pq[0].lb_value if pq else np.inf), stats=stats, message=msg)

    # Global LB is PQ head if any left, else last known LB
    final_lb = pq[0].lb_value if pq else global_lb
    return BnBResult(True, best_x, float(best_obj), float(final_lb), stats,
                     message=f"Solved to gap {best_obj - final_lb:.3g} in {stats.time_sec:.2f}s")
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

# -----------------------------------------------------------------
# __main__
# -----------------------------------------------------------------

if __name__ == "__main__":
    # Example: run on your JSON config
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--time", type=float, default=60.0)
    args = p.parse_args()

    n,m,c,H,Q,A,b,lb,ub = _load_problem_from_json(args.config)

    def log(msg): print(msg)

    opts = BnBOptions(
        abs_gap_tol=1e-4, rel_gap_tol=1e-3,
        max_nodes=3000, time_limit_sec=args.time,
        ub_extra_starts=40, ub_maxiter=4000,
        strong_branch_topk=3, prepartition_pieces=3,
        enable_implied_bounds=True
    )

    res = branch_and_bound_advanced(n,m,c,H,Q,A,b,lb,ub, opts=opts, logger=log)

    print("\n=== Advanced B&B Result ===")
    print(res.message)
    print(f"Success: {res.success}")
    print(f"Best UB: {res.obj}")
    print(f"Global LB: {res.global_lb}")
    if res.obj is not None and np.isfinite(res.global_lb):
        print(f"Abs gap: {res.obj - res.global_lb:.6g}")
        print(f"Rel gap: {(res.obj - res.global_lb)/(1+abs(res.obj)):.6g}")
    print(f"Nodes: gen={res.stats.nodes_generated}, exp={res.stats.nodes_expanded}, pruned={res.stats.nodes_pruned}")
    if res.x is not None:
        print("x* =", res.x)
