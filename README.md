# Asatro

**A** **S**ophisticated (or **S**imple) **A**pproach **T**o f**R**agment gr**O**wing.

Asatro grows elaborations from a **bound fragment** in its crystallographic pose,
choosing what to make and where with **Thompson Sampling** over reaction × reactant
space, and placing each candidate by constrained docking onto the original pose.

It grew out of — and has since superseded — the TS+GNINA combinatorial web app
(`/opt/webapps/TS`, now decommissioned): Asatro's fragment-anchored growth path
was split out of it, and its reaction catalog, RWS sampler, and
protecting-group handling have since been folded back in here.

Asatro is:

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

Working end-to-end and real-dock validated.

**Tier-1 handle detection + auto-core derivation** (`asatro/chemistry/`). Given a
fragment it reports the functional-group handles it bears, the compatible start
reactions (and which slot the fragment fills), and the conserved core
auto-derived for each handle.

```bash
python -m asatro.chemistry.handles "OC(=O)c1ccncc1"   # CLI
curl 'http://localhost:5015/analyze?smiles=OC(=O)c1ccncc1'
python -m pytest tests/                                # 155 passing
```

**Reaction catalog**: the full Hartenfeller et al. reaction SMIRKS set (58
reactions — amide/Suzuki/SNAr/reductive-amination/Buchwald-Hartwig/Negishi/
Sonogashira/Heck/Wittig/Mitsunobu/urea/thiourea and a dozen heterocycle
syntheses: Pictet-Spengler, benzimidazole/-oxazole/-thiazole, quinazoline,
pyrazole, Paal-Knorr pyrrole, Fischer indole, Friedländer quinoline,
benzofuran/-thiophene, indole and more), 53 functional-group classes
(`asatro/data/reactions.json`, `asatro/data/functional_groups.json`,
`asatro/chemistry/catalog.py`), ported from the reference set checked in at
[`Hartenfeller_reactions.txt`](Hartenfeller_reactions.txt):

> Hartenfeller, M.; Eberle, M.; Meier, P.; Nieto-Oberhuber, C.; Altmann, K.-H.;
> Schneider, G.; Jacoby, E.; Renner, S. *A Collection of Robust Organic
> Synthesis Reactions for In Silico Molecule Design.* J. Chem. Inf. Model.
> **2011**, 51 (12), 3093–3098.

Every reaction is a single catalog row — any reaction can serve as step 1
*or* extend a running intermediate at one of its own slots (`resolve_step`),
so there's no separate hand-authored start/extend duplication. Browse the
full set — id, components, accepted reagent classes, full SMARTS — at
`GET /reactions`, linked from the app header.

Ported verbatim except one confirmed bug in the reference file itself:
`decarboxylative_coupling`'s `!OH1` is a SMARTS scoping mistake (`!` binds
only to `O`, so it means "not-oxygen, 1 H" rather than the clearly-intended
"not a hydroxyl"), which excluded every real substrate — including the
reference's own example reagent — from matching at all. Fixed the negation
scoping; nothing else touched. (Asatro's own prior 54-reaction, ts-gnina-
derived catalog is retired — this session found and fixed 6 similar
regiochemistry/atom-loss bugs in it before switching to the validated
reference set wholesale instead of continuing to audit our own port.)

Growth mode's placement guard (below) assumes a reacting handle's local
geometry survives largely intact into the product (a leaving group departs,
everything else stays anchored) — that holds for substitution/coupling-type
reactions but not for the ring-forming ones in the set (Pictet-Spengler,
benzimidazole/-thiazole/-oxazole, Paal-Knorr, Fischer indole, etc.), where the
reacting site's hybridization changes outright. Those fire and dock fine in
**combi** (unanchored) mode; in growth mode they'll typically fail the
core-RMSD guard for most candidates — a mismatch between "anchor a known
binding mode" and "build a new ring around the anchor atom," not a bug.

The **accessibility pre-pass** is in too. A fast geometric cone probe
(`accessibility.py`) measures how far each growth vector reaches before hitting
receptor atoms; an optional **stub-growth refinement** (`stub_growth.py`) then
grows real –Me/–Ph/morpholine substituents onto the survivors, constrained to the
bound pose, and keeps only vectors where a substituent physically fits.

```bash
# fragment SDF (in its bound pose) + receptor PDB -> analysis + accessible reactions
curl -F fragment=@hit.sdf -F receptor=@receptor.pdb http://localhost:5015/prune
curl -F fragment=@hit.sdf -F receptor=@receptor.pdb -F refine=true \
     http://localhost:5015/prune          # + stub-growth refinement
```

Two search paths share the same lifted Thompson-Sampling + GNINA stack
(`asatro/engine/`: `AnchoredFragmentEvaluator`, `GninaEvaluator`, `RouteSampler`, …):

- **Fragment growth** (`asatro/growth.py`): pick one accessible reaction/slot as
  step 1 — the bound fragment fills that slot — and optionally chain further
  "extend" steps onto it; each final product is constrained-placed onto the
  bound pose and scored by GNINA, with an adjustable core-RMSD placement guard.
  The accessibility pre-pass validates the chosen route before it runs.
- **Combinatorial search** (`asatro/combi.py`): the same route-building/search
  machinery with no bound fragment — every slot is a real reagent library,
  freely embedded and docked (matching ts-gnina's own search).

Either path is Thompson-Sampled by default, or Roulette Wheel Selection with
thermal cycling (Zhao et al. 2025) for better coverage on 3+-component routes.
Every product is stripped of Boc/Cbz/Fmoc/ester/boronate protecting groups
before docking and reporting — commercial building blocks commonly carry one
on a handle other than the one reacted, so the deliverable compound, not the
protected intermediate, is what gets scored.

Both run as **background jobs**:

```bash
curl -F fragment=@hit.sdf -F receptor=@receptor.pdb \
     -F reactants=@boronic.smi \              # one .smi per FG-class slot
     -F 'config={"steps":["suzuki"],"fragment_slot":1,"num_cycles":50,"refine":true}' \
     http://localhost:5015/grow               # -> {"job_id": ...}
curl http://localhost:5015/jobs/<id>          # status + top hits
curl http://localhost:5015/jobs/<id>/stream   # live console (SSE)
```

(The dock needs the `gnina` binary at `/opt/gnina/gnina.1.3.2` + a GPU; everything
else runs anywhere.)

A **browser UI** (`templates/index.html`, served at `/`) drives the whole flow —
Fragment growth and Combinatorial search as two modes: upload inputs → *Analyze*
(fragment growth) or build a route (combi) → configure reagents/filters/search →
launch, with a live SSE console, structure gallery + convergence chart, and a
job-history picker. Dark/light theme.

Still open: conflict-aware pool tagging (a difunctional block currently lands in
every matching class) and persisted/curated pools. See [DESIGN.md](DESIGN.md).

## Setup

```bash
conda env create -f environment.yml   # creates the `asatro` env
conda activate asatro
./run.sh                              # serves on http://0.0.0.0:${ASATRO_PORT:-5015}
```

The `asatro` conda env already exists on this host (python 3.11, rdkit, fastapi,
uvicorn, openbabel). Docking will additionally need the `gnina` binary
(see `/opt/webapps/gnina`).

## Deployment (systemd + ufw)

Runs as a systemd service on **port 5015**, reachable on the KTH network at
`http://130.237.250.75:5015` (this host's IP; the app binds `0.0.0.0`). Mirrors the
TS+GNINA app's setup. Requires `sudo`.

```bash
# Install + start the service (asatro-webapp.service is in this repo)
sudo cp /opt/webapps/asatro/asatro-webapp.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now asatro-webapp
sudo systemctl status asatro-webapp        # check it came up
```

```bash
# Firewall: allow the port from the KTH network only (same convention as ts-gnina)
sudo ufw allow from 130.237.0.0/16 to any port 5015 proto tcp \
     comment 'Asatro fragment-growing webapp (KTH only)'
```
# asatro
