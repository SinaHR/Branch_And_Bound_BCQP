# BCQP Node Bounds and Branch & Bound Solver

This project implements a **Branch & Bound global optimization framework** for **Bilinearly Constrained Quadratic Programs (BCQPs)** that contain **bilinear (nonconvex)** equality constraints.

It provides:

* Convex **McCormick relaxations** for lower bounds (LBs)
* Multi-start local optimization for upper bounds (UBs)
* Two versions of a full **Branch & Bound** algorithm

  * A simple, educational variant
  * An advanced version with bound tightening and strong branching

---

## Problem Formulation

The solver handles BCQPs of the form:

**Objective:**

```
minimize    f(x) = xᵀ H x + cᵀ x
```

**Subject to equality constraints:**

```
gₖ(x) = xᵀ Qₖ x + Aₖ x - bₖ = 0,    for k = 1, …, m
```

**And bound constraints:**

```
lb ≤ x ≤ ub
```

where:

* `H` is a symmetric matrix defining the quadratic objective
* Each `Q_k` is **symmetric with zero diagonal**, so only bilinear terms (`x_i * x_j`, with `i ≠ j`) appear
* All variable bounds (`lb`, `ub`) must be **finite**, required for the McCormick relaxation to be valid.

---

## Components

### 1. `node_lower_bound(...)`

Computes a **convex relaxation** of theBCQP using **McCormick envelopes** to replace bilinear terms.
Returns a provable lower bound (LB) on the global optimum.

**Features**

* Automatically identifies bilinear pairs from (Q_k) matrices.
* Constructs linearized envelope inequalities.
* Solves via `scipy.optimize.minimize(method='SLSQP')`.
* Checks if (H) is positive semidefinite (convex objective).
* Returns a detailed `NodeLBResult` dataclass.

---

### 2. `node_upper_bound(...)`

Solves the **original nonconvex BCQP** using local optimization and multiple random restarts to find feasible points and upper bounds (UBs).

**Features**

* Uses analytic Jacobians for speed.
* Employs random restarts within [lb, ub] and optional warm-start.
* Returns the best feasible solution found within tolerance.
* Output stored in `NodeUBResult`.

---

### 3. `node_bounds_pipeline(...)`

Convenience wrapper that runs `node_lower_bound` → `node_upper_bound` sequentially and reports:

* LB (relaxed value)
* UB (best feasible value)
* Absolute and relative gaps

Output: `NodeBoundsResult`.

---

### 4. `branch_and_bound_simple(...)` – Simple B&B

Implements a basic **best-bound** Branch & Bound algorithm that:

1. Solves the relaxation (LB) at each node
2. Computes a UB from a feasible point
3. Branches on the variable with the largest width in bilinear pairs
4. Prunes nodes with LB ≥ best UB
5. Stops when the gap is within tolerance

Output: `BnBResult` with stats (LB, UB, gaps, nodes, time).

---

### 5. `branch_and_bound_advanced(...)` – Advanced B&B

A research-grade version with stronger bounding and smarter branching.

**Enhancements**

* **Implied-bound tightening (OBBT)** at each node
* **Degree × width** variable selection
* **Strong branching** probes among top-K candidates
* **Root pre-partitioning** for piecewise McCormick tightening
* **Warm-start upper bounds**
* Structured `BnBOptions` and `BnBStats`

Output: `BnBResult` with timing and node statistics.

---

## Input Format (JSON)

Each problem is defined in a `.json` file:

```json
{
  "n": 3,
  "m": 2,
  "c": [1, 1, 0],
  "b": [8, 15],
  "A": [[0,0,1],[0,1.8,0]],
  "H": [[0,0,0],[0,0,0],[0,0,1]],
  "Q": [
    [[0,0.5,0],[0.5,0,0.5],[0,0.5,0]],
    [[0,0.5,0],[0.5,0,0],[0,0,0]]
  ],
  "lb": [0, 0, 0],
  "ub": [10, 10, 10]
}
```

Notes:

* `Q` can be a list of matrices or a 3D array.
* Diagonal entries in all `Q_k` must be **zero**.
* All bounds must be **finite**.

---

## Usage

### Run via CLI

#### 1. Run the **Branch & Bound solver**

```bash
python branch_and_bound_advanced.py --config INPUT.json
```

#### 2. Run the **McCormick relaxation and bounds pipeline**

```bash
python mccormick_relaxation.py --config INPUT.json --pipeline
```

---

## Command-Line Options

Each main script supports `--help` to display available arguments.

### `branch_and_bound_advanced.py`

```
usage: branch_and_bound_advanced.py --config INPUT.json [--time TIME_LIMIT]

options:
  --config INPUT.json     Path to JSON input file defining the QCQP problem.
  --time TIME_LIMIT       Optional time limit in seconds for the Branch & Bound search.
```

### `mccormick_relaxation.py`

```
usage: mccormick_relaxation.py --config INPUT.json [--pipeline | --lb-only | --ub-only] [options]

modes:
  --pipeline               Run both lower-bound and upper-bound routines sequentially (default).
  --lb-only                Solve only the McCormick relaxation.
  --ub-only                Solve only the non-relaxed QCQP for a feasible UB.

lower-bound options:
  --lb-verbose             Print optimizer progress.
  --lb-maxiter N           Maximum SLSQP iterations (default 2000).
  --lb-no-analytic-jacobians  Disable analytic Jacobians for LB solve.
  --lb-sym-tol EPS         Tolerance for symmetry check (default 1e-8).

upper-bound options:
  --ub-extra-starts N      Number of random restarts (default 20).
  --ub-rng-seed SEED       RNG seed (default 42).
  --ub-tol-eq EPS          Feasibility tolerance (default 1e-6).
  --ub-maxiter N           Max SLSQP iterations (default 2000).
  --ub-verbose             Print UB solver progress.
```

---

## Output Interpretation

| Term                 | Description                                      |
| -------------------- | ------------------------------------------------ |
| **UB (Upper Bound)** | Best feasible objective found so far             |
| **LB (Lower Bound)** | Best relaxation objective (provable lower bound) |
| **Gap**              | ( UB - LB ); smaller gap = closer to optimal     |
| **Nodes expanded**   | Subproblems solved                               |
| **Nodes pruned**     | Branches discarded by bounding                   |
| **LB / UB solves**   | Number of subproblem optimizations performed     |

> A **large gap** means the relaxation is loose or the search is shallow.
> A **small gap** (≈0) means the algorithm has nearly certified the global optimum.

---

## Algorithmic Notes

* **Convexity Detection:**
  The LB routine checks if (H) is positive semidefinite.
  If not, the LB is a *candidate* from a local solve (still valid but not globally tight).

* **Bilinear-only Constraints:**
  Each (Q_k) must have a zero diagonal.
  Diagonal terms would introduce non-bilinear quadratic terms not supported here.

* **OBBT Bound Tightening:**
  Uses rows with one linear variable to shrink domains before solving each node.

* **Strong Branching:**
  Probes top-K candidate variables to choose the split that raises bounds fastest.

---

## Example Test Files

| File        | Description                                |
| ----------- | ------------------------------------------ |
| `test.json` | 3-variable nonconvex test with wide bounds |

---

## Dependencies

Install all requirements:

```bash
pip install -r requirements.txt
```

---

## Summary Table

| Component                   | Role                   | Type             | Solver            |
| --------------------------- | ---------------------- | ---------------- | ----------------- |
| `node_lower_bound`          | Convex relaxation      | Lower bound      | SLSQP             |
| `node_upper_bound`          | Original BCQP          | Upper bound      | Multi-start SLSQP |
| `branch_and_bound_simple`   | Basic global search    | B&B              | Simple            |
| `branch_and_bound_advanced` | Enhanced global search | B&B + Tightening | Advanced          |

---

## Visual Summary

```
 ┌────────────────────────┐
 │  Root QCQP problem     │
 └────────────┬───────────┘
              │
     Relax (LB) via McCormick
              │
     Local solve (UB) via SLSQP
              │
        Branch variable
        /              \
   Child 1           Child 2
      ↓                 ↓
  Repeat LB/UB       Repeat LB/UB
              │
     Update global LB/UB
              │
     Stop when gap small
```
