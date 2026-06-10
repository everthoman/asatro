# Asatro — design

## Goal

Given a **bound fragment** (a hit in its crystallographic pose) and its receptor,
propose and rank synthetically reasonable **elaborations that grow the fragment
into open pocket space**, while keeping the known binding mode fixed.

## Pipeline

```
bound fragment (SDF, in pose) + receptor
        │
        ▼
1. Handle / vector detection   ── what can chemistry do to this scaffold?
        │
        ▼
2. Accessibility pruning       ── which growth directions have room in the pocket?
        │
        ▼
3. Thompson-Sampled growth     ── search reaction × reactant library, constrained-
   + constrained placement        place onto the bound pose, score by GNINA
        │
        ▼
ranked elaborations
```

### 1. Handle / vector detection

- **Tier 1 — existing handles (have a prototype):** tag the fragment against a
  functional-group SMARTS vocabulary; a start reaction is *compatible* when one of
  its components accepts a class the fragment bears. The reacting handle defines a
  conserved core (fragment minus the leaving atoms), auto-derived per FG class via
  a `leaving_smarts` rule. Prototyped in the TS repo against the same vocabulary.
- **Tier 2 — installable handles (ambitious):** positions with no handle but an
  obvious **growth vector** (an aromatic C–H to halogenate/borylate, a derivatizable
  position). "Proposable reactions" = existing handles ∪ installable vectors. This
  is what makes Asatro feel like it reasons about the scaffold rather than reading
  off a functional group.

### 2. Accessibility pruning ("prune samples to find inaccessible areas")

Each candidate reaction implies a **growth vector** — the bond that extends from the
conserved core. Before spending search budget, screen vectors for pocket room:

- **Geometric probe (fast, approximate) — IMPLEMENTED** (`accessibility.py`): from
  the bound pose, cast a cone of directions along each exit vector and march each
  out until it hits a receptor atom (`free` distance, capped at `max_reach`). A
  vector is accessible if its best direction clears `min_free`; survivors carry
  `free_central / mean_free / max_free / open_fraction` for ranking. Tunable via
  `ProbeParams`. Exposed as `POST /prune` (fragment SDF + receptor PDB).
- **Stub-growth sampling (slower, reliable) — IMPLEMENTED** (`stub_growth.py`): for
  each surviving vector, build core + stub (–Me/–Ph/morpholine), constrained-embed
  with the core pinned to the bound pose (coordMap + Kabsch realign), and keep only
  vectors where *some* stub places without clashing the receptor. Reports which
  stubs fit (and the largest) as a richness cue. Runs only on geometric survivors
  (`assess_with_stubs`); exposed as `POST /prune` with `refine=true`. Note: the
  geometric clash check is already strict on-axis, so the stub pass mainly adds
  fidelity for bulky/planar groups and multi-atom reach.

Net effect: the main search only ever explores growable directions.

### 3. Thompson-Sampled growth + constrained placement — ENGINE LIFTED (`growth.py`, `engine/`)

- Each reactant slot is a bandit arm; Thompson Sampling docks only an adaptively
  chosen subset of products, converging on good regions of a (potentially
  REAL-scale) reactant library — the key edge over enumerate-and-place.
- Placement = constrained embed onto the fragment's bound coordinates +
  local-only docking + a core-drift guard (`AnchoredFragmentEvaluator`, lifted
  into `asatro/engine/`).
- Scoring = GNINA (CNN optional); the conserved core is pinned so the score
  reflects how well the *elaboration* extends the known mode, not a free re-dock.
- Wiring: the bound fragment is fixed in one start-reaction slot (a one-entry
  reagent file); `RouteSampler` samples the other slot(s). `growth.run_growth()`
  builds the route, the anchored evaluator, and runs warm-up + search. Verified up
  to constrained placement (a biphenyl grown onto a bound benzene); the dock
  itself needs the gnina binary (`/opt/gnina/gnina.1.3.2`) + a GPU.
- **Pre-pass → growth connection** (`grow_accessible` / `plan_targets`): the
  accessibility assessment gates the search — only accessible reaction/slots become
  growth targets, each carrying its auto-derived conserved core; non-fragment
  components are resolved to reactant files via an injectable resolver. So pruned
  vectors never cost a docking call.
- **Job/endpoint layer** (`jobs.py`, `app.py`): a growth run is a background thread
  (`start_growth_job`) that runs the pre-pass then grows the survivors, streaming
  console lines and persisting a per-target results summary + metadata under the
  job dir. Endpoints: `POST /grow` (fragment SDF + receptor PDB + reactant .smi
  files named by FG class + JSON config), `GET /jobs`, `GET /jobs/{id}`,
  `POST /jobs/{id}/cancel`, `GET /jobs/{id}/stream` (SSE). The docking runner is
  injectable, so the whole flow is tested without gnina.
- TODO: curated reactant libraries for the non-fragment slots, and a real gnina
  dock run (binary + GPU) to validate scoring; a browser UI on top of the routes.

## Differentiation summary (vs Syndirella)

| | Syndirella | Asatro |
|---|---|---|
| Direction | retrosynthetic (decompose an analogue) | forward (handles on the bound hit) |
| Library coverage | enumerate then place | Thompson-Sampled subset |
| Pocket awareness | scoring-time | explicit pre-pass pruning of dead vectors |
| Placement | Fragmenstein | constrained embed + local docking + drift guard |

## Open decisions (resolve when building)

- **Tier-2 installable handles** in v1, or Tier-1 only first?
- **Accessibility pruning**: geometric-only (fast) vs include stub-growth (reliable)?
  Lean: Tier-1 + stub-growth first — already beats the retro/enumerate paradigm.
- **Synthesizability**: rely on curated reactant sets, or add a retro-feasibility
  filter on products? (TS edge is sampling; retro could be a post-filter.)

## Reuse (lift later, deliberately — not copied yet)

From `/opt/webapps/TS`: `anchored_fragment_evaluator.py` (constrained placement +
drift guard, already pocket-anchored), `gnina_evaluator.py`/`evaluators.py`
(docking + overridable `_prepare_pose`/`_extra_flags` hooks), the TS sampler stack
(`thompson_sampling.py`, `route_sampler.py`, `reagent.py`, `ts_utils.py`,
`disallow_tracker.py`), and the FG vocabulary + `reactions.json`.
