# Drone facility location computational supplement

This package contains the source code, generated input instances, solution arrays,
computational records, and figure scripts for **A Hybrid Lagrangian-Relaxation Framework for Drone-Based Facility Location Decisions **.

## Installation

Python 3.10 or later is required. From this directory:

```bash
python -m pip install -r code/requirements.txt
```

SciPy/HiGHS is sufficient for the compact model, linear programs, and verification.
Gurobi comparisons additionally require `gurobipy` and an appropriate Gurobi license.
Gurobi itself and its license are not distributed in this package.

## Reproduce tables and figures from saved results

These commands read saved inputs and solutions. They do not run an optimization
experiment.

```bash
python code/analyse_results.py --root .
python code/plot_paper.py --root . --output figures
python code/verify_package.py --root .
python code/audit_drone_allocations.py --root .
```

`tables/analysis.json` and the CSV tables contain the recomputed numerical values.
Figures 2–10 are written as PNG and vector PDF files. Figure 1 is the conceptual
illustration accompanying the manuscript. `results/package_audit.json` contains
the feasibility, input-fingerprint, objective, and bound-order checks.
`results/drone_allocation_audit.json` checks individual-drone payload, energy,
home-field uniqueness, and consistency with the saved aggregate flows.

## Directory structure

| Path | Contents |
| --- | --- |
| `code/flpd_revision.py` | Exact count model, facility relaxation, labelled formulation, reconstruction and audits |
| `code/flpd_core_v2.py` | Lagrangian search, feasible recovery and facility local search |
| `code/flpd_instances_v2.py` | Original instance generator |
| `code/run_revision.py` | Experiment driver and per-task persistence |
| `code/run_geography.py` | Optional documented-coordinate input interface |
| `code/check_legacy_sensitivity.py` | Independent count-model checks of archived resource inputs |
| `code/audit_legacy_open_sets.py` | Transportation-LP reconstruction and equal-share audits of archived field sets |
| `code/analyse_results.py` | Manuscript table calculations and audited battery reuse |
| `code/plot_paper.py` | Figure generation from stored numerical records |
| `code/summarize_revision.py` | Raw-record summaries and archived benchmark statistics |
| `code/verify_package.py` | Saved-solution verification |
| `results/benchmark` | Equal-budget comparison and extended labelled-model references |
| `results/bounds_short` | 30 s compact-model references for 60 and 100 customers |
| `results/ablation_small` | Nine-input component study |
| `results/steps_small` | Three-input multiplier study |
| `results/sites_small` | 36 candidate-site designs |
| `results/battery_exact_check` | Independent count-model battery checks |
| `results/payload_exact_check` | Direct count-model payload checks |
| `results/reconstructed_legacy_sweeps` | Independently reconstructed allocations for archived resource field sets |
| `results/evidence` | LP diagnostics and fifty cross-capacity certificates |
| `results/local_fleet` | Priced, bounded and two-type fleet records |
| `results/local_perturb` | Fixed-field recourse and field-reselection scenarios |
| `results/count45` | Independent compact-model checks and LP values |
| `results/validation` | Count-versus-label validation inputs and outcomes |
| `data/legacy` | Archived benchmark and resource-sensitivity source files |
| `tables` | Recomputed analytical tables |
| `figures` | Publication figures |

The directory names of executed experiments are retained so that saved record
identities and configuration hashes remain traceable.
`CODE_VERSIONS.json` maps recorded numerical code fingerprints to the release
source and `code/executed_v1`. The archived version was used for initial HiGHS
reference checks. The release adds nonfinite-value sanitization at JSON output;
the numerical optimization routines are unchanged between those versions.

## Optional numerical replication

Use a fresh output directory for each parameter configuration. The following
command runs the exact-count and labelled validation cases with HiGHS:

```bash
python code/run_revision.py --suite validate --backend highs --mip-seconds 30 --output replication/validation
```

A compact-model reference for one generated input:

```bash
python code/run_revision.py --suite bounds --sizes 30 --layouts C --seeds 202601 --backend highs --mip-seconds 15 --output replication/count_example
```

The component study used:

```bash
python code/run_revision.py --suite ablation --sizes 30 --layouts C R RC --seeds 202601 202602 202603 --iterations 200 --lr-seconds 30 --mip-seconds 30 --backend gurobi --output replication/ablation
```

The multiplier and candidate-site studies used:

```bash
python code/run_revision.py --suite steps --sizes 30 --layouts C R RC --seeds 202601 --iterations 200 --lr-seconds 30 --mip-seconds 30 --backend gurobi --output replication/steps
python code/run_revision.py --suite sites --sizes 30 --layouts C R RC --seeds 202601 202602 202603 --mip-seconds 15 --backend gurobi --output replication/sites
```

To reproduce the complete equal-budget comparison, select `--suite benchmark`
with `--sizes 30 --full-label`. The saved extended references used
`--extended-seconds 3600`; those runs can require one hour per labelled instance.
Extended solves are unnecessary for regenerating any table or figure. Default
size lists include more inputs, so specify `--sizes`, `--layouts` and `--seeds`
explicitly when running a bounded replication.

`--resume` skips completed tasks only when the saved code and configuration
fingerprints match. Existing archived directories should be treated as read-only;
use `replication/` for new runs. The driver sets numerical library thread limits
before loading NumPy/SciPy. Gurobi uses one solver thread and seed zero.

## Model and numerical interpretation

Demand is continuously divisible aggregate mass. Each drone has one home field;
payload and round-trip energy are cumulative planning-period resources. The base
objective contains field-opening and demand-weighted transportation costs. The
count formulation is exact under these assumptions: each field–type flow is
divided equally among its integer number of drones. It is not an exact formulation
for indivisible parcels or a sortie-scheduling problem.

The facility-only model omits drone resources and supplies a lower bound. A solver
incumbent from this relaxation is not itself an original-model feasible plan.
Agg-Pack reconstructs and audits a plan before accepting an upper bound. The LR
dual equals the original model's LP relaxation; its finite-search bound can be
weaker. The count MILP supplies an independent exact-model incumbent and bound.

Battery budgets use model energy units with synthetic coefficient 1.43. No
drone-specific physical calibration is assigned. Drone cost 20 is a synthetic
cost scenario. Free-fleet drone counts are not minimum-fleet certificates.

Optimal status, time-limit status, missing incumbents, and proved infeasibility
are distinct. Only solver bounds are lower bounds from time-limited exact models.
The certified-gap denominator is the feasible objective; objective excess over a
known optimum uses the optimum as its denominator.

## Experimental provenance

`benchmark`, `bounds_short`, `ablation_small`, `steps_small`, and `sites_small`
contain the Windows/Gurobi records used for the controlled comparison and
additional studies. The benchmark folder also contains one completed 60-customer
sequence; that input is excluded from the fifteen-input comparison in Table 3.
The 60- and 100-customer results in Table 2 use `bounds_short`, consistently across
all fifteen inputs at each size.

The independent battery, payload, fleet and recourse checks use SciPy/HiGHS.
Their wall times are not pooled with the controlled Gurobi comparison. Original
45-input LR-H results are preserved in `data/legacy`; historical and current
runtime records are not treated as cross-machine speed comparisons. Configuration
files retain the environment information available at execution. Some optional
defaults listed in those configurations were not selected by the active suite.

Battery certificates in `tables/battery_reference.csv` use the best compatible
count incumbent or independently reconstructed field allocation only after a
fresh target-budget audit. Lower bounds combine valid count-model bounds and
the archived facility-only bound. A less restrictive battery problem's lower
bound remains valid for a more restrictive problem. This independent envelope
is separate from direct LR outcomes and the LR-only reusable envelope. The
LR-only battery envelope is recomputed from the original objectives and saved
maximum battery loads; its source records remain unchanged.

Every payload cross-capacity certificate has a saved target input and audited
solution array. An independently feasible capacity-3 allocation matches a valid
facility bound and remains feasible at capacities 4, 5, 6 and 8. Earlier positive
payload-related changes in heuristic objectives are not optimal-cost effects.


