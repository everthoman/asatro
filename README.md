# Asatro

**A** **S**ophisticated (or **S**imple) **A**pproach **T**o f**R**agment gr**O**wing.

Asatro grows elaborations from a **bound fragment** in its crystallographic pose,
choosing what to make and where with **Thompson Sampling** over reaction × reactant
space, and placing each candidate by constrained docking onto the original pose.

It is a sibling of — but deliberately distinct from — the TS+GNINA combinatorial
web app (`/opt/webapps/TS`), which Asatro's fragment functionality was split out of.

## Why it's different from Syndirella

Syndirella elaborates a base compound **retrosynthetically and exhaustively**:
decompose a known analogue, pull purchasable reactant analogues, enumerate the
whole elaboration library, place every member (Fragmenstein), score. Cost scales
with the enumerated library.

Asatro instead is:

1. **Forward & handle-driven** — it reads the reactive handles *on the bound
   fragment itself* and proposes the reactions those handles enable; no
   pre-existing analogue to retrosynthesize.
2. **Thompson-Sampled, not enumerate-and-place** — each reactant slot is a bandit;
   only a small, adaptively chosen subset is ever docked, so REAL-scale reactant
   sets are tractable.
3. **Pocket-aware** — a cheap accessibility pre-pass prunes growth vectors that can
   only grow into protein walls *before* the search spends its docking budget.

See [DESIGN.md](DESIGN.md) for the full pipeline and open decisions.

## Status

Early, but the first engine slice is in: **Tier-1 handle detection + auto-core
derivation** (`asatro/chemistry/`). Given a fragment it reports the functional-group
handles it bears, the compatible start reactions (and which slot the fragment
fills), and the conserved core auto-derived for each handle.

```bash
python -m asatro.chemistry.handles "OC(=O)c1ccncc1"   # CLI
curl 'http://localhost:5023/analyze?smiles=OC(=O)c1ccncc1'
python -m pytest tests/                                # 17 passing
```

The **accessibility pre-pass** is in too. A fast geometric cone probe
(`accessibility.py`) measures how far each growth vector reaches before hitting
receptor atoms; an optional **stub-growth refinement** (`stub_growth.py`) then
grows real –Me/–Ph/morpholine substituents onto the survivors, constrained to the
bound pose, and keeps only vectors where a substituent physically fits.

```bash
# fragment SDF (in its bound pose) + receptor PDB -> analysis + accessible reactions
curl -F fragment=@hit.sdf -F receptor=@receptor.pdb http://localhost:5023/prune
curl -F fragment=@hit.sdf -F receptor=@receptor.pdb -F refine=true \
     http://localhost:5023/prune          # + stub-growth refinement
```

Still to build: the Thompson-Sampled growth + constrained placement + GNINA
scoring. That will reuse pieces from the TS repo
(`anchored_fragment_evaluator.py`, the TS sampler stack), lifted in deliberately.
See [DESIGN.md](DESIGN.md).

## Setup

```bash
conda env create -f environment.yml   # creates the `asatro` env
conda activate asatro
./run.sh                              # serves on http://0.0.0.0:${ASATRO_PORT:-5023}
```

The `asatro` conda env already exists on this host (python 3.11, rdkit, fastapi,
uvicorn, openbabel). Docking will additionally need the `gnina` binary
(see `/opt/webapps/gnina`).
# asatro
