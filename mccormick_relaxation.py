from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import json
import argparse
import numpy as np
import numpy.typing as npt
from scipy import optimize
import warnings

# =========================
# Data containers
# =========================

@dataclass
class NodeLBResult:
    """Result container for the lower-bound (relaxed problem) solve."""
    res: optimize.OptimizeResult          # Full optimization result object from scipy
    x: np.ndarray                         # Solution vector for original decision variables
    y: Optional[np.ndarray]               # Solution vector for helper (bilinear) variables
    bilinear_terms: List[Tuple[int, int]] # List of index pairs representing bilinear terms
    Aeq: np.ndarray                       # Equality constraint matrix after relaxation
    beq_rhs: np.ndarray                   # Right-hand side of equality constraints
    B: Optional[np.ndarray]               # Inequality constraint matrix (McCormick envelope)
    b_ineq: Optional[np.ndarray]          # Right-hand side for inequality constraints
    was_convex: bool                     # True iff H was PSD under the tolerance
    psd_tol: float                       # Tolerance used in the check

@dataclass
class NodeUBResult:
    """Result container for the upper-bound (original bilinear-equality QCQP) solve."""
    success: bool                   # True if a feasible solution (within tol) was found
    x: Optional[np.ndarray]         # Best feasible x found (None if none found)
    obj: Optional[float]            # Objective value at x (None if none found)
    violation: float                # Max absolute equality violation at x
    result: Optional[optimize.OptimizeResult]  # SciPy result object of the winning run
    start_index: Optional[int]      # Which start produced the best feasible point

@dataclass
class NodeBoundsResult:
    """Container holding both the lower-bound relaxation result and the upper-bound bilinear-equality QCQP result."""
    lb_res: "NodeLBResult"                 # result from node_lower_bound(...)
    ub_res: "NodeUBResult"                 # result from node_upper_bound(...)
    lb_value: float                        # objective from LB relaxation (lower bound)
    ub_value: Optional[float]              # best feasible objective (upper bound), or None if not found
    gap_abs: Optional[float]               # |UB - LB| if UB exists
    gap_rel: Optional[float]               # (UB - LB) / (1 + |UB|) or similar, if UB exists

# -----------------------------------------------------------------------------
# Helper function to validate dimensions and check symmetry of Q matrices
# -----------------------------------------------------------------------------

def _assert_shapes(n: int, m: int,
                   c: np.ndarray, H: np.ndarray, Q: np.ndarray,
                   A: np.ndarray, b: np.ndarray, lb: np.ndarray, ub: np.ndarray,
                   *, sym_tol: float = 1e-8) -> None:
    """Validate dimensions and symmetry. Q matrices must be symmetric and have ZERO diagonal (bilinear-only)."""
    if Q.ndim != 3 or Q.shape != (n, n, m):
        raise ValueError(f"Q must have shape (n,n,m)=({n},{n},{m}), got {Q.shape}")
    if H.shape != (n, n):
        raise ValueError(f"H must be (n,n)=({n},{n}), got {H.shape}")
    if A.shape != (m, n):
        raise ValueError(f"A must be (m,n)=({m},{n}), got {A.shape}")
    if c.shape != (n,):
        raise ValueError("c must be a 1-D vector of length n")
    if b.shape != (m,):
        raise ValueError("b must be a 1-D vector of length m")
    if lb.shape != (n,) or ub.shape != (n,):
        raise ValueError("lb and ub must be 1-D vectors of length n")
    if np.any(lb > ub):
        raise ValueError("lb > ub for some variables")

    # H symmetric
    if not np.allclose(H, H.T, atol=sym_tol, rtol=sym_tol*10):
        raise ValueError("H must be symmetric within tolerance.")

    # Each Q_k symmetric AND strictly zero diagonal (within tolerance)
    for k in range(m):
        Qi = Q[:, :, k]
        if not np.allclose(Qi, Qi.T, atol=sym_tol, rtol=sym_tol*10):
            raise ValueError(f"Q[:,:,{k}] is not symmetric within tolerance {sym_tol}")
        if np.any(np.abs(np.diag(Qi)) > sym_tol):
            raise ValueError(f"Q[:,:,{k}] has nonzero diagonal; only bilinear (no x_i^2) terms allowed.")

# -----------------------------------------------------------------------------
# Lower-bound solver for the relaxed problem (McCormick envelopes)
# -----------------------------------------------------------------------------
def node_lower_bound(n: int,
                     m: int,
                     c: List[float],
                     H: npt.NDArray[np.float64],
                     Q: npt.NDArray[np.float64],
                     A: npt.NDArray[np.float64],
                     b: List[float],
                     lb: List[float],
                     ub: List[float],
                     *,
                     provide_analytic_jacobians: bool = True,
                     maxiter: int = 2000,
                     verbose: bool = False,
                     sym_tol: float = 1e-8) -> NodeLBResult:

    # Convert input lists to numpy arrays for consistent operations
    c = np.asarray(c, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    lb = np.asarray(lb, dtype=float).reshape(-1)
    ub = np.asarray(ub, dtype=float).reshape(-1)
    H = np.asarray(H, dtype=float)
    Q = np.asarray(Q, dtype=float)
    A = np.asarray(A, dtype=float)

    # Validate shapes and symmetry
    _assert_shapes(n, m, c, H, Q, A, b, lb, ub, sym_tol=sym_tol)

    # Warn only here (LB path) if H is not PSD => relaxation not truly convex
    w = np.linalg.eigvalsh(H)
    psd_tol = 1e-10 * max(1.0, float(np.max(np.abs(w))))
    was_convex = bool(w.min() >= -psd_tol)
    if not was_convex:
        warnings.warn(
            "H is not positive semidefinite; the McCormick-relaxed problem is "
            "not globally convex. The returned value is a candidate lower bound "
            "(SLSQP local solution) rather than the tightest convex relaxation.",
            category=UserWarning,
            stacklevel=2
        )

    # Collect unique pairs of indices that represent bilinear terms
    bilinear_terms: List[Tuple[int, int]] = []
    seen: Dict[Tuple[int, int], int] = {}

    # Extract upper-triangular indices for each Q matrix to find bilinear relationships
    triu = np.triu_indices(n)
    for k in range(m):
        Qi = Q[:, :, k]
        vals = Qi[triu]
        mask = np.abs(vals) > sym_tol
        rows = triu[0][mask]
        cols = triu[1][mask]
        for i, j in zip(rows, cols):
            if i == j:
                raise ValueError("Quadratic (diagonal) terms detected; only bilinear allowed.")
            if (i, j) not in seen:
                seen[(i, j)] = len(bilinear_terms)
                bilinear_terms.append((i, j))
    
    for (i1, i2) in bilinear_terms:
        if not (np.isfinite(lb[i1]) and np.isfinite(ub[i1]) and np.isfinite(lb[i2]) and np.isfinite(ub[i2])):
            raise ValueError(
            f"Finite bounds required for McCormick on variables {i1} and {i2}."
        )
    # Count the number of helper variables required
    count = len(bilinear_terms)

    # Build extended equality constraint matrix (Aeq)
    if count > 0:
        A_help = np.zeros((m, count), dtype=float)
        for k in range(m):
            Qi = Q[:, :, k]
            vals = Qi[triu]
            mask = np.abs(vals) > sym_tol
            rows = triu[0][mask]
            cols = triu[1][mask]
            for i, j in zip(rows, cols):
                pos = seen[(i, j)]
                A_help[k, pos] += 2.0 * Qi[i, j]  # off-diagonal term counted twice
        Aeq = np.hstack((A, A_help))  # combine A with helper variable columns
    else:
        A_help = np.zeros((m, 0))
        Aeq = A

    # Construct McCormick inequality constraints for bilinear terms
    if count > 0:
        B = np.zeros((4 * count, n + count), dtype=float)
        b_ineq = np.zeros(4 * count, dtype=float)
        r = 0
        for idx, (i1, i2) in enumerate(bilinear_terms):
            y = n + idx  # helper variable index
            l1, u1 = lb[i1], ub[i1]
            l2, u2 = lb[i2], ub[i2]
            # Four linear inequalities defining the convex envelope
            B[r, y], B[r, i2], B[r, i1], b_ineq[r] = -1, l1, l2, l1 * l2; r += 1
            B[r, y], B[r, i2], B[r, i1], b_ineq[r] = -1, u1, u2, u1 * u2; r += 1
            B[r, y], B[r, i2], B[r, i1], b_ineq[r] = 1, -u1, -l2, -u1 * l2; r += 1
            B[r, y], B[r, i2], B[r, i1], b_ineq[r] = 1, -l1, -u2, -l1 * u2; r += 1
    else:
        B, b_ineq = None, None

    # Extend objective matrices and vectors with zeros for helper variables
    if count > 0:
        H_relax = np.block([[H, np.zeros((n, count))], [np.zeros((count, n)), np.zeros((count, count))]])
        c_relax = np.concatenate((c, np.zeros(count)))
    else:
        H_relax, c_relax = H, c

    # Define initial guess: midpoint of bounds for x and zeros for helper vars
    x0 = np.concatenate(((lb + ub) / 2.0, np.zeros(count)))

    # Define quadratic objective function and its gradient
    def obj(z):
        return float(z @ (H_relax @ z) + c_relax @ z)

    def grad(z):
        return 2.0 * (H_relax @ z) + c_relax

    # Assemble equality and inequality constraints for optimization
    constraints = []

    # Equality constraint: nonlinear equation approximations
    def ceq(z, Aeq=Aeq, b=b):
        return b - Aeq @ z

    if provide_analytic_jacobians:
        constraints.append({'type': 'eq', 'fun': ceq, 'jac': lambda z, Aeq=Aeq: -Aeq})
    else:
        constraints.append({'type': 'eq', 'fun': ceq})

    # Inequality constraints (McCormick relaxation)
    if count > 0 and B is not None and b_ineq is not None:
        def cineq(z, B=B, b_ineq=b_ineq):
            return b_ineq - B @ z
        if provide_analytic_jacobians:
            constraints.append({'type': 'ineq', 'fun': cineq, 'jac': lambda z, B=B: -B})
        else:
            constraints.append({'type': 'ineq', 'fun': cineq})

    # Variable bounds (original + helper)
    bounds = optimize.Bounds(np.concatenate((lb, np.full(count, -np.inf))), np.concatenate((ub, np.full(count, np.inf))))

    # Solve relaxation using SLSQP
    res = optimize.minimize(obj, x0, method='SLSQP', jac=grad if provide_analytic_jacobians else None,
                            bounds=bounds, constraints=constraints, options={'disp': verbose, 'maxiter': maxiter})
    
    # ---------------------------------------------------------------------
    # ON THE "LOWER BOUND" RELAXATION:
    #
    # We relax the (bilinear) equality constraints using McCormick envelopes,
    # which yields convex constraint sets in the extended (x, y) space.
    # The overall problem is convex IF AND ONLY IF the quadratic objective
    # is convex, i.e., H is positive semidefinite (H ⪰ 0).
    #
    # • If H ⪰ 0: solving the relaxation globally gives a true convex
    #   relaxation; its global optimum is a valid (tightest) lower bound.
    # • If H is indefinite: the objective is nonconvex; the relaxation is
    #   still a valid constraint relaxation (the feasible set is enlarged),
    #   so the GLOBAL optimum still provides a lower bound, but the problem
    #   itself is nonconvex. Using SLSQP (local) yields only a candidate LB.
    # -----------------------------------------------------------------------

    # Extract optimized variables and organize results
    z_opt = np.asarray(res.x, dtype=float)
    x_opt = z_opt[:n]
    y_opt = z_opt[n:] if count > 0 else None

    # Return structured result with all relevant matrices and solver output
    return NodeLBResult(res=res, x=x_opt, y=y_opt, bilinear_terms=bilinear_terms, Aeq=Aeq, beq_rhs=b, B=B, b_ineq=b_ineq,
                         was_convex=was_convex,psd_tol=psd_tol)


# -----------------------------------------------------------------------------
# Upper-bound solver for the original (non-relaxed) problem
# -----------------------------------------------------------------------------

def node_upper_bound(n: int,
                     m: int,
                     c: List[float],
                     H: npt.NDArray[np.float64],
                     Q: npt.NDArray[np.float64],
                     A: npt.NDArray[np.float64],
                     b: List[float],
                     lb: List[float],
                     ub: List[float],
                     *,
                     x0: Optional[np.ndarray] = None,   # optional warm-start (e.g., from a heuristic)
                     extra_starts: int = 20,            # number of random restarts in [lb, ub]
                     rng_seed: Optional[int] = 42,      # RNG seed for reproducibility
                     tol_eq: float = 1e-6,              # feasibility tolerance on equalities
                     maxiter: int = 2000,
                     verbose: bool = False) -> NodeUBResult:
    """Compute an *upper bound* by solving the original non-relaxed bilinear-equality QCQP.

    Objective:  minimize  f(x) = x^T H x + c^T x
    s.t.        g_k(x) = x^T Q_k x + A_k x - b_k = 0,   for k=0..m-1
                lb <= x <= ub

    This routine uses `scipy.optimize.minimize` with exact Jacobians for speed, and
    performs multiple random restarts within the variable bounds to reduce the risk
    of getting stuck in a poor local optimum. The best *feasible* solution found
    (within `tol_eq`) is returned as an upper bound.
    """
    # Convert to arrays
    c = np.asarray(c, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    lb = np.asarray(lb, dtype=float).reshape(-1)
    ub = np.asarray(ub, dtype=float).reshape(-1)
    H = np.asarray(H, dtype=float)
    Q = np.asarray(Q, dtype=float)
    A = np.asarray(A, dtype=float)

    # Basic validation (reuse the same checks)
    _assert_shapes(n, m, c, H, Q, A, b, lb, ub)

    # If you want to fail fast:
    if not (np.all(np.isfinite(lb)) and np.all(np.isfinite(ub))):
        raise ValueError(
            "UB random restarts require finite lb/ub to sample initial points. "
            "Provide finite bounds or set extra_starts=0 and pass a custom x0."
        )

    # Precompute bounds and helpers
    bounds = optimize.Bounds(lb, ub)

    # Objective and gradient
    def f_obj(x: np.ndarray) -> float:
        return float(x @ (H @ x) + c @ x)

    def f_grad(x: np.ndarray) -> np.ndarray:
        # gradient of x^T H x + c^T x is (H + H^T) x + c; H is symmetric -> 2 H x + c
        return 2.0 * (H @ x) + c

    # Equality constraints g(x) = 0 and Jacobian (m-vector)
    def g_eq(x: np.ndarray) -> np.ndarray:
        # g_k(x) = x^T Q_k x + A_k x - b_k
        # Compute for all k in one pass
        vals = np.empty(m, dtype=float)
        for k in range(m):
            Qk = Q[:, :, k]
            vals[k] = float(x @ (Qk @ x) + A[k, :] @ x - b[k])
        return vals

    def g_jac(x: np.ndarray) -> np.ndarray:
        # Jacobian matrix (m x n): row k is grad g_k(x) = (Q_k + Q_k^T) x + A_k^T = 2 Q_k x + A_k^T
        J = np.empty((m, n), dtype=float)
        for k in range(m):
            Qk = Q[:, :, k]
            J[k, :] = 2.0 * (Qk @ x) + A[k, :]
        return J

    # Vector-valued equality constraint for SciPy (SLSQP supports vector constraints)
    nl_eq = {'type': 'eq', 'fun': g_eq, 'jac': g_jac}

    # Build starting points: optional warm start + uniform random in box
    starts: List[np.ndarray] = []
    if x0 is not None:
        x0 = np.clip(np.asarray(x0, dtype=float).reshape(-1), lb, ub)
        if x0.size != n:
            raise ValueError("x0 must have length n")
        starts.append(x0)
    rng = np.random.default_rng(rng_seed)
    for _ in range(extra_starts):
        starts.append(rng.uniform(lb, ub))

    # Track the best feasible solution found
    best_x: Optional[np.ndarray] = None
    best_obj: Optional[float] = None
    best_res: Optional[optimize.OptimizeResult] = None
    best_violation: float = np.inf
    best_idx: Optional[int] = None

    # Solve from each start
    for idx, x_start in enumerate(starts):
        res = optimize.minimize(
            f_obj, x_start, method='SLSQP', jac=f_grad,
            bounds=bounds, constraints=[nl_eq],
            options={'disp': verbose, 'maxiter': maxiter}
        )
        x_candidate = np.asarray(res.x, dtype=float)
        # Evaluate feasibility
        viol = float(np.max(np.abs(g_eq(x_candidate))))

        if viol <= tol_eq:
            # Feasible candidate: consider as an upper bound
            val = f_obj(x_candidate)
            if (best_obj is None) or (val < best_obj):
                best_x = x_candidate
                best_obj = val
                best_res = res
                best_violation = viol
                best_idx = idx

    success = best_x is not None
    return NodeUBResult(success=success,
                        x=best_x,
                        obj=(best_obj if success else None),
                        violation=(best_violation if success else np.inf),
                        result=best_res,
                        start_index=best_idx)

# --- Example usage (commented) ---
# if __name__ == "__main__":
#     n = 3; m = 2
#     c = [1, 1, 0]
#     b = [8, 15]
#     A = np.zeros((m, n)); A[0, 2] = 1
#     H = np.zeros((n, n)); H[2, 2] = 1
#     Q1 = np.zeros((n, n)); Q1[0, 1] = Q1[1, 0] = 0.5
#     Q2 = np.zeros((n, n)); Q2[1, 2] = Q2[2, 1] = 0.5
#     Q = np.dstack((Q1, Q2))
#     lb = [0, 0, 0]; ub = [10, 10, 10]
#     ub_res = node_upper_bound(n, m, c, H, Q, A, b, lb, ub, extra_starts=30, verbose=True)
#     print(ub_res)


# =========================
# Pipeline
# =========================

def node_bounds_pipeline(
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
    lb_verbose: bool = False,
    lb_maxiter: int = 2000,
    lb_provide_analytic_jacobians: bool = True,
    lb_sym_tol: float = 1e-8,
    ub_extra_starts: int = 20,
    ub_rng_seed: Optional[int] = 42,
    ub_tol_eq: float = 1e-6,
    ub_maxiter: int = 2000,
    ub_verbose: bool = False,
) -> NodeBoundsResult:
    lb_res = node_lower_bound(
        n, m, c, H, Q, A, b, lb, ub,
        provide_analytic_jacobians=lb_provide_analytic_jacobians,
        maxiter=lb_maxiter, verbose=lb_verbose, sym_tol=lb_sym_tol
    )
    lb_value = float(lb_res.res.fun)
    x0 = np.clip(lb_res.x, np.asarray(lb, dtype=float), np.asarray(ub, dtype=float))
    ub_res = node_upper_bound(
        n, m, c, H, Q, A, b, lb, ub,
        x0=x0, extra_starts=ub_extra_starts, rng_seed=ub_rng_seed,
        tol_eq=ub_tol_eq, maxiter=ub_maxiter, verbose=ub_verbose
    )
    ub_value = ub_res.obj if ub_res.success else None
    if ub_value is not None:
        gap_abs = float(abs(ub_value - lb_value))
        gap_rel = float((ub_value - lb_value) / (1.0 + abs(ub_value)))
    else:
        gap_abs = None
        gap_rel = None
    return NodeBoundsResult(lb_res=lb_res, ub_res=ub_res,
                            lb_value=lb_value, ub_value=ub_value,
                            gap_abs=gap_abs, gap_rel=gap_rel)

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

# =========================
# __main__
# =========================

if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    n, m, c, H, Q, A, b, lb, ub = _load_problem_from_json(args.config)

    # LB only?
    if args.lb_only:
        lb_res = node_lower_bound(
            n, m, c, H, Q, A, b, lb, ub,
            provide_analytic_jacobians=(not args.lb_no_analytic_jacobians),
            maxiter=args.lb_maxiter, verbose=args.lb_verbose, sym_tol=args.lb_sym_tol
        )
        print("=== Lower Bound ===")
        print(f"LB value: {float(lb_res.res.fun)} "
              f"{'' if lb_res.was_convex else '(nonconvex objective; candidate LB)'}")
        print("x (LB):", lb_res.x)
        print("y (LB):", lb_res.y)
        exit(0)

    # UB only?
    if args.ub_only:
        # Optionally warm-start UB from the midpoint (simple) if desired:
        x0 = (np.asarray(lb) + np.asarray(ub)) / 2.0
        ub_res = node_upper_bound(
            n, m, c, H, Q, A, b, lb, ub,
            x0=x0, extra_starts=args.ub_extra_starts,
            rng_seed=args.ub_rng_seed, tol_eq=args.ub_tol_eq,
            maxiter=args.ub_maxiter, verbose=args.ub_verbose
        )
        print("=== Upper Bound ===")
        if ub_res.success:
            print("UB value:", ub_res.obj)
            print("x (UB):", ub_res.x)
            print("max |eq violation|:", ub_res.violation)
        else:
            print("No feasible solution found within tolerance.")
        exit(0)

    # Default: Pipeline (LB -> UB)
    out = node_bounds_pipeline(
        n, m, c, H, Q, A, b, lb, ub,
        lb_verbose=args.lb_verbose, lb_maxiter=args.lb_maxiter,
        lb_provide_analytic_jacobians=(not args.lb_no_analytic_jacobians),
        lb_sym_tol=args.lb_sym_tol,
        ub_extra_starts=args.ub_extra_starts, ub_rng_seed=args.ub_rng_seed,
        ub_tol_eq=args.ub_tol_eq, ub_maxiter=args.ub_maxiter, ub_verbose=args.ub_verbose
    )
    print("=== Pipeline Results ===")
    print(f"Lower bound (relaxation): {out.lb_value} "
      f"{'' if out.lb_res.was_convex else '(nonconvex objective; candidate LB)'}")
    if out.ub_value is not None:
        print(f"Upper bound (bilinear-equality QCQP):       {out.ub_value}")
        print(f"Abs gap: {out.gap_abs}, Rel gap: {out.gap_rel}")
        print("Best feasible x:", out.ub_res.x)
        print("Max |eq violation|:", out.ub_res.violation)
    else:
        print("Upper bound: not found (no feasible point within tolerance)")

