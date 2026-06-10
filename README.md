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
curl 'http://localhost:5015/analyze?smiles=OC(=O)c1ccncc1'
python -m pytest tests/                                # 17 passing
```

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

The **TS growth engine** is lifted in too. `asatro/engine/` holds standalone
copies of the Thompson-Sampling + GNINA stack (`AnchoredFragmentEvaluator`,
`RouteSampler`, …); `asatro/growth.py` wires them up: the bound fragment is fixed
in one start-reaction slot, the reactant library is sampled in the other(s), and
each product is constrained-placed onto the bound pose and scored by GNINA. The
accessibility pre-pass gates it — only surviving reaction/slots are searched.

Growth runs as a **background job**:

```bash
curl -F fragment=@hit.sdf -F receptor=@receptor.pdb \
     -F reactants=@boronic.smi \              # one .smi per FG-class slot
     -F 'config={"num_cycles":50,"refine":true}' \
     http://localhost:5015/grow               # -> {"job_id": ...}
curl http://localhost:5015/jobs/<id>          # status + per-target top hits
curl http://localhost:5015/jobs/<id>/stream   # live console (SSE)
```

(The dock needs the `gnina` binary at `/opt/gnina/gnina.1.3.2` + a GPU; everything
else runs anywhere.)

A **browser UI** (`templates/index.html`, served at `/`) drives the whole flow:
upload the bound fragment + receptor → *Analyze accessibility* (shows detected
handles, per-reaction accessible/pruned status, and each auto-core) → upload a
`.smi` library per surviving building-block slot → *Launch growth*, with a live
SSE console, per-target results tables, and a job-history picker. Dark/light theme.

Still to build: curated reactant libraries for the non-fragment slots, and a real
gnina dock run to validate scoring. See [DESIGN.md](DESIGN.md).

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
