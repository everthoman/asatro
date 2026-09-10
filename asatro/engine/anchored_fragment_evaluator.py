"""
Anchored fragment-growing evaluator. Wired into the web app via
``asatro.growth.make_evaluator`` / ``run_growth``; real-dock validated against
`gnina.1.3.2` + GPU (see DESIGN.md).

Use case
--------
You have a fragment whose *bound pose* is known (crystal soak / reliable dock).
You grow it combinatorially over a 1- or 2-step reaction route (fragment as a
one-member reagent set; bifunctional BBs added at the exit vector) and want to
rank the grown products by *how well the growth extends the known binding mode*
-- NOT by a free re-dock that is allowed to flip the whole molecule into an
unrelated pose.

How it differs from GninaEvaluator
----------------------------------
GninaEvaluator builds each product with a *free* ETKDGv3 embed and lets gnina do
a *global* search in the box. Both steps discard the fragment's known pose. This
subclass overrides just those two steps:

1. 3D build  -> ``rdkit.Chem.AllChem.ConstrainedEmbed`` onto the reference
   fragment's 3D coordinates, so the conserved core starts at (and is
   restrained to) its bound position while only the grown part is embedded.
2. Docking   -> a normal gnina search, but in a box sized to *this candidate's*
   anchored conformer (see ``_box_flags``), so the search is confined to the
   fragment's own site instead of the whole protein.

Plus two post-dock guards, applied per docked mode (``_pose_acceptable``):

* the conserved-core atoms must not have drifted more than ``max_core_rmsd`` A
  from the reference -- this is what holds the binding mode, since the search
  itself is free; ``None`` switches it off (drift is still measured and
  annotated, never rejected).
* the pose must not be jammed into the receptor: no heavy atom within
  ``clash_radius`` A of a receptor atom, and ``minimizedAffinity`` (gnina's
  empirical score, always computed) not above ``max_affinity``.

Why this protocol, measured rather than assumed. The constrained embed places
the grown arm without ever seeing the receptor, so its starting pose is usually
inside the protein: over ten products, the closest heavy-atom approach of the
built pose was 0.57-2.22 A, with up to seven atoms under 1.8 A. The old
protocol handed exactly that pose to ``--local_only`` -- a local optimisation,
no search -- and asked it to cope. Sometimes it does (on a small anchor it
pushed all ten of those back out to 2.3-3.4 A), but it has only two ways out of
a bad start, and both were seen in production: settle into the clash, which on
one 877-product run left 356 poses at positive ``minimizedAffinity`` (up to
+32 kcal/mol, physically impossible), or slide the whole ligand out of the site,
which is what that run's top-ranked pose did -- 12.5 A off its anchor. Neither
is visible through a CNN score field: CNN_VS and minimizedAffinity were
uncorrelated there (r = 0.03) and 39 of its top 50 by CNN_VS scored positive.
That protocol is gone rather than optional: it was never worth its speed.

A real search in the same box is better on every axis measured: on ten products
it beat local-only's affinity on nine (by 0.3-4.0 kcal/mol) and stayed anchored
on all of them, where local-only drifted past 2 A on three; over a full
278-product run every pose came back at -5.2 to -8.4 kcal/mol with its closest
heavy-atom contact at 2.70-3.06 A. It costs ~7x the time per dock (2.4 s ->
15-40 s), which is the price of a pose worth looking at.

Note what this means for the clash guard: under a real search it is close to
inert (zero rejections over that 278-product run, no pose within 2.5 A), since
the search optimises the very term a clash violates. It is kept as the backstop
for the one case a search cannot signal -- a product that fits nowhere still
comes back as ``num_modes`` best-effort poses, never as "no pose".

Required refactor seam in GninaEvaluator (tiny, behaviour-preserving)
---------------------------------------------------------------------
``GninaEvaluator._dock`` currently inlines ligand prep and the flag list. To
subclass cleanly, factor those two out into overridable hooks (defaults keep the
present behaviour):

    # in GninaEvaluator._dock, replace
    #     sdf_block, err = prepare_ligand_3d(smiles, self.ph, "ligand")
    # with
    #     sdf_block, err = self._prepare_pose(smiles)
    # and append self._extra_flags() to the cmd list.

    def _prepare_pose(self, smiles):            # default = current behaviour
        return prepare_ligand_3d(smiles, self.ph, "ligand")

    def _extra_flags(self):                     # default = none
        return []

This module assumes those two hooks exist.
"""

from __future__ import annotations

import multiprocessing
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

from asatro.chemistry.accessibility import rotate_about, torsion_axis, load_receptor_atoms
from asatro.chemistry.handles import carve_substructure_3d, neutralize
from asatro.engine.gnina_evaluator import (
    GninaEvaluator,
    prepare_ligand_3d,
    protonate_smiles,
    _strip_sdf_properties,
)


def _match_core(frag: Chem.Mol, core_str: str):
    """Find the conserved core in ``frag``. ``core_str`` may be SMARTS or SMILES;
    we try both, then an aromaticity-tolerant fallback, so the user doesn't have
    to know which the field wants. Returns ``(match_tuple, query_mol)`` (match is
    () when nothing hits)."""
    queries = []
    qs = Chem.MolFromSmarts(core_str)
    if qs is not None:
        queries.append(qs)
    qm = Chem.MolFromSmiles(core_str)        # users often paste SMILES
    if qm is not None:
        queries.append(qm)
    if not queries:
        raise ValueError(f"core could not be parsed as SMARTS or SMILES: '{core_str}'")
    for q in queries:
        m = frag.GetSubstructMatch(q)
        if m:
            return m, q
    # Aromaticity-tolerant retry: kekulize both sides and match without the
    # aromatic-flag constraint (catches kekulized-vs-aromatic mismatches).
    try:
        frag_k = Chem.Mol(frag)
        Chem.Kekulize(frag_k, clearAromaticFlags=True)
        for q in queries:
            qk = Chem.Mol(q)
            try:
                Chem.Kekulize(qk, clearAromaticFlags=True)
            except Exception:
                pass
            m = frag_k.GetSubstructMatch(qk)
            if m:
                return m, q
    except Exception:
        pass
    return (), queries[0]


def _movable_core_atoms(frag: Chem.Mol, match: Sequence[int]) -> Tuple[int, ...]:
    """Core atoms whose *position* the bound pose does not fix, as indices into
    the carved core.

    The reaction consumes the leaving group, and the handle it hangs off can
    turn about the single bond into the core -- a carboxyl flips its -OH and its
    C=O, an aryl sulfonyl chloride spins its S=O pair. Those atoms survive into
    the product but their bound coordinates are one arbitrary torsion state, so
    pinning them would force the new substituent onto whichever site the leaving
    group happened to occupy (see DESIGN.md; the same reasoning as the
    accessibility pass's rotamers). Everything else in the core is genuinely
    conserved.
    """
    core_of = {f: c for c, f in enumerate(match)}
    leaving = {a.GetIdx() for a in frag.GetAtoms()
               if a.GetIdx() not in core_of and a.GetAtomicNum() != 1}
    for attach in match:
        a = frag.GetAtomWithIdx(attach)
        if not any(n.GetIdx() in leaving for n in a.GetNeighbors()):
            continue
        anchor = torsion_axis(frag, attach, leaving)
        if anchor is None:
            continue
        movable = tuple(sorted(
            core_of[n.GetIdx()] for n in a.GetNeighbors()
            if n.GetIdx() in core_of and n.GetIdx() != anchor
            and n.GetAtomicNum() != 1 and n.GetDegree() == 1))
        if movable:
            return movable
    return ()


def _flipped_core(core: Chem.Mol, movable: Tuple[int, ...]) -> Optional[Chem.Mol]:
    """The core template with its ``movable`` atoms turned 180 deg about the bond
    into the rest of the core -- the other torsion state the handle can adopt,
    so the product can be built with the new bond on either site."""
    if not movable:
        return None
    conf = core.GetConformer()
    pos = lambda i: np.array(conf.GetAtomPosition(i))
    attach = None
    for a in core.GetAtomWithIdx(movable[0]).GetNeighbors():
        attach = a.GetIdx()
        break
    if attach is None:
        return None
    anchor = torsion_axis(core, attach, set(movable))
    if anchor is None:
        return None
    axis = pos(attach) - pos(anchor)
    n = float(np.linalg.norm(axis))
    if n < 1e-6:
        return None
    flipped = Chem.Mol(core)
    fconf = flipped.GetConformer()
    turned = rotate_about(np.array([pos(i) for i in movable]), pos(attach), axis / n, 180.0)
    for i, xyz in zip(movable, turned):
        fconf.SetAtomPosition(i, Point3D(*(float(v) for v in xyz)))
    return flipped


def _load_core(fragment_sdf: str, core_smarts: Optional[str]) -> Tuple[Chem.Mol, Tuple[int, ...]]:
    """
    Build the conserved-core template (a 3D mol) used both to seed the embed and
    to check pose drift, plus the indices of its atoms whose position the bound
    pose does not actually fix (``_movable_core_atoms``).

    ``fragment_sdf`` is the fragment in its *bound* pose. ``core_smarts``, if
    given, selects the sub-part of the fragment that survives the growth
    reaction unchanged -- crucial because the reactive handle changes on
    reaction (an acid's -OH leaves, an aryl-Br's Br leaves), so the leaving atom
    is NOT part of the product and must be excluded from the match template.
    If omitted, the whole fragment heavy-atom graph is used (correct only when
    no atoms are lost, e.g. SNAr onto a ring C-F where F is replaced 1:1... in
    practice almost always pass an explicit core_smarts).
    """
    frag = Chem.MolFromMolFile(fragment_sdf, removeHs=True)
    if frag is None:
        raise ValueError(f"Could not read fragment SDF: {fragment_sdf}")
    if frag.GetNumConformers() == 0:
        raise ValueError("Fragment SDF has no 3D conformer (need the bound pose)")
    # Bound poses are often protonated (e.g. a primary amine as [NH3+]), but
    # core_smarts is derived from the neutralized fragment (matching handles.py's
    # analyze_fragment) and every reaction product has that atom neutral post-
    # reaction. Neutralize here too so the carved core's charge state actually
    # matches what it needs to substruct-match against.
    frag = neutralize(frag)
    if core_smarts is None:
        return frag, ()
    match, q = _match_core(frag, core_smarts)
    if not match:
        frag_smiles = Chem.MolToSmiles(frag)
        raise ValueError(
            f"core_smarts '{core_smarts}' does not match the fragment "
            f"(fragment from SDF = '{frag_smiles}'). The core must be a "
            f"substructure of the bound fragment; check aromaticity (e.g. use "
            f"aromatic 'c1ccncc1', not kekulized 'C1=CC=NC=C1') and that you "
            f"excluded only the reacting handle, not ring atoms.")
    # Carve the matched atoms out *with their coordinates* into a core template.
    try:
        return carve_substructure_3d(frag, match), _movable_core_atoms(frag, match)
    except ValueError as e:
        # More specific context than the generic carving error: here we know
        # it was core_smarts, not an arbitrary match, that cut the ring.
        raise ValueError(
            f"the conserved core carved from the fragment is not a valid "
            f"substructure ({e}). This usually means core_smarts excluded an "
            f"in-ring atom -- exclude only the reacting handle (the leaving "
            f"atom/group), and keep whole aromatic rings intact.") from e


_EMBED_TIMEOUT_DEFAULT = 60  # seconds

# Post-dock pose guards. The core-RMSD default is 2.0 A because the docking is a
# real search now: a clean, correctly-anchored pose from a free search sits ~1.2
# to 1.8 A off the reference core (measured over a re-docked production run),
# where the old local-only protocol left it under 1.0 A by construction -- the
# same 1.5 A that once meant "drifted" would now reject good poses.
DEFAULT_MAX_CORE_RMSD = 2.0
# A heavy atom this close to a receptor atom is through the surface, not in
# contact with it (a real H-bond heavy-atom pair sits at ~2.7-3.2 A).
DEFAULT_CLASH_RADIUS = 1.8
# minimizedAffinity is gnina's empirical (Vina-like) score, computed for every
# pose whatever the score field is. Above zero the steric term has won: the
# pose is repulsive on the physics the CNN does not see.
DEFAULT_MAX_AFFINITY = 0.0

# ``forkserver``, not the platform default ``fork`` or ``spawn``:
# - ``fork`` from a process that has other live threads (the concurrent-dock
#   ThreadPoolExecutor workers calling this) is fragile -- only the forking
#   thread survives in the child, and any lock held by a non-forking thread
#   at the moment of fork stays permanently "held" in the child, a classic
#   source of child-side deadlocks.
# - ``spawn`` re-imports/re-executes the *calling program's* __main__ module
#   in every child (that's how it reconstructs enough state to unpickle the
#   target) -- fine for a script that guards its top level with
#   ``if __name__ == "__main__":``, but this is library code with no control
#   over the caller. Confirmed the hard way: an unguarded test script's
#   top-level `start_growth_job(...)` call got re-executed inside every
#   single spawned embed worker, each of which then spawned its own
#   recursive embed workers, each re-executing the script again.
# ``forkserver`` starts one single-threaded helper process (safe to fork
# from) a single time, up front -- new workers are forked from *that*, not
# from this multi-threaded caller and not by re-running __main__ per call.
_MP_CTX = multiprocessing.get_context("forkserver")


def _embed_worker(mol_bytes: bytes, core_bytes: bytes, seed: int, out_q) -> None:
    """Runs in an isolated child process (see ``_run_constrained_embed``)."""
    try:
        mol = Chem.Mol(mol_bytes)
        core = Chem.Mol(core_bytes)
        AllChem.ConstrainedEmbed(mol, core, randomseed=seed)
        out_q.put(("ok", mol.ToBinary()))
    except Exception as e:  # noqa: BLE001 -- report back, don't crash the child silently
        out_q.put(("error", str(e)))


def _run_constrained_embed(mol: Chem.Mol, core: Chem.Mol, seed: int, timeout: float) -> Chem.Mol:
    """``AllChem.ConstrainedEmbed``, isolated in its own process and bounded by
    a wall-clock timeout.

    RDKit's embedding has no native timeout and, for a strained or very
    flexible growth, can occasionally take pathologically long or never
    converge -- with no way to interrupt the C++ call mid-flight from Python.
    An earlier version of this ran the embed in a worker *thread* and simply
    stopped waiting on timeout, but that doesn't free anything: Python
    threads can't be force-killed, so the abandoned computation kept running
    (and, on the one real production case that triggered this, kept
    allocating memory without bound -- ~48GB RSS and climbing, one stuck
    candidate after another, no docking subprocess ever even started). A
    *process* can actually be killed and its memory reclaimed by the OS, so
    that's what a timeout does here."""
    q = _MP_CTX.Queue()
    p = _MP_CTX.Process(target=_embed_worker, args=(mol.ToBinary(), core.ToBinary(), seed, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join(5)
        if p.is_alive():
            p.kill()
            p.join()
        raise TimeoutError(
            f"constrained embed exceeded {timeout}s (likely a strained/pathological "
            f"conformer) -- worker process killed")
    try:
        # A short blocking get, not get_nowait(): multiprocessing.Queue hands
        # data to the parent via a background feeder thread over a pipe, so
        # even after a normal (non-timeout) exit there's a brief window where
        # p.join() has returned but the item hasn't landed in the queue yet.
        status, payload = q.get(timeout=5)
    except Exception:
        raise RuntimeError("constrained embed worker exited without a result "
                           "(likely crashed/OOM in the child process)")
    if status == "error":
        raise RuntimeError(payload)
    return Chem.Mol(payload)


def _receptor_clashes(mol: Chem.Mol, receptor: np.ndarray, radius: float = 2.2) -> int:
    """Heavy atoms of ``mol`` sitting inside ``radius`` of a receptor atom."""
    if receptor.size == 0 or mol.GetNumConformers() == 0:
        return 0
    conf = mol.GetConformer()
    xyz = np.array([list(conf.GetAtomPosition(a.GetIdx())) for a in mol.GetAtoms()
                    if a.GetAtomicNum() != 1])
    if xyz.size == 0:
        return 0
    near = receptor[np.linalg.norm(receptor - xyz.mean(axis=0), axis=1) <= 25.0]
    if near.size == 0:
        return 0
    d2 = ((near[None, :, :] - xyz[:, None, :]) ** 2).sum(axis=2)
    return int((d2.min(axis=1) < radius * radius).sum())


def _receptor_contacts(mol: Chem.Mol, receptor: np.ndarray) -> Optional[float]:
    """Closest heavy-atom approach between ``mol`` and the receptor, in A.
    ``None`` when there is nothing to measure. Annotated onto every pose: it is
    the one number that says "this is in the pocket" or "this is through the
    wall", and unlike a score it means the same thing in every run."""
    if receptor.size == 0 or mol.GetNumConformers() == 0:
        return None
    conf = mol.GetConformer()
    xyz = np.array([list(conf.GetAtomPosition(a.GetIdx())) for a in mol.GetAtoms()
                    if a.GetAtomicNum() != 1])
    if xyz.size == 0:
        return None
    near = receptor[np.linalg.norm(receptor - xyz.mean(axis=0), axis=1) <= 25.0]
    if near.size == 0:
        return None
    d2 = ((near[None, :, :] - xyz[:, None, :]) ** 2).sum(axis=2)
    return float(np.sqrt(d2.min()))


def _constrained_pose_block(
    smiles: str, ph: float, core: Chem.Mol, seed: int = 0xF00D,
    embed_timeout: float = _EMBED_TIMEOUT_DEFAULT,
    alt_cores: Sequence[Chem.Mol] = (), receptor: Optional[np.ndarray] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Protonate ``smiles`` (reusing GninaEvaluator's obabel step), then build a 3D
    pose with the ``core`` atoms pinned at their bound coordinates via
    ConstrainedEmbed. Returns ``(sdf_block, error)`` shaped like prepare_ligand_3d.

    ``alt_cores`` are the same core in the other torsion states its handle can
    adopt (see ``_flipped_core``). They are tried only if the as-posed template
    puts the product into the receptor, and the variant with the fewest clashes
    wins -- so a fragment whose leaving group faced a wall grows out the other
    way instead of being built into it. Without a ``receptor`` there is nothing
    to choose on, so the as-posed template is used as before.
    """
    # Reuse the existing protonate step (same OpenBabel -p pH call the free
    # combi path uses) so a basic amine on the grown building block -- not
    # part of the conserved core, which stays neutral, see _load_core -- comes
    # out charged here too, instead of docking (and reporting hits as) the
    # neutral tautomer. Formal charge isn't part of RDKit's default
    # Mol-vs-Mol substructure match (verified: an [NH3+] target still matches
    # a neutral "N" query), so protonating before the core match/embed below
    # doesn't put the two out of step.
    protonated_smiles, prot_err = protonate_smiles(smiles, ph)
    if protonated_smiles is None:
        return None, prot_err
    mol = Chem.MolFromSmiles(protonated_smiles)
    if mol is None:
        return None, f"RDKit could not parse protonated '{protonated_smiles}' (from '{smiles}')"
    mol = Chem.AddHs(mol)
    if not mol.HasSubstructMatch(core):
        # The conserved core is not present -> the reaction did not preserve the
        # fragment (wrong route / wrong exit vector). Reject like a prep failure.
        return None, "conserved fragment core not found in product"
    templates = [core] + [c for c in alt_cores if c is not None]
    best = best_clashes = None
    for template in templates:
        try:
            # ConstrainedEmbed: matches core in mol, fixes those atoms at the core
            # coordinates, embeds the rest, and runs a restrained MMFF minimisation.
            # Runs isolated in a child process (see _run_constrained_embed) -- the
            # embedded conformer comes back on `mol`, not mutated in place.
            placed = _run_constrained_embed(mol, template, seed, embed_timeout)
        except Exception as e:  # embedding can fail (or time out) for very strained grows
            if template is templates[-1] and best is None:
                return None, f"constrained embed failed: {e}"
            continue
        if receptor is None or len(templates) == 1:
            best = placed
            break
        clashes = _receptor_clashes(placed, receptor)
        if best is None or clashes < best_clashes:
            best, best_clashes = placed, clashes
        if best_clashes == 0:
            break
    if best is None:
        return None, "constrained embed failed for every core orientation"
    mol = best
    block = Chem.MolToMolBlock(mol) + "$$$$\n"
    lines = block.split("\n")
    if lines:
        lines[0] = "ligand"
    return _strip_sdf_properties("\n".join(lines)), None


class AnchoredFragmentEvaluator(GninaEvaluator):
    """
    GninaEvaluator that grows from a *bound* fragment: constrained embed onto the
    fragment pose, a gnina search boxed to that candidate, and post-dock guards
    on core drift and receptor clash.

    Extra ``input_dict`` keys (on top of GninaEvaluator's)
        fragment_sdf : str        - fragment in its bound pose (3D SDF). REQUIRED.
                                     Also satisfies GninaEvaluator's "give me a
                                     site" requirement, though the box itself is
                                     no longer sized from it -- see _box_flags.
        core_smarts  : str        - the conserved sub-fragment (exclude the
                                     leaving handle). Strongly recommended.
        max_core_rmsd: float (2.0)- reject if core drifts more than this (A).
                                     ``None`` turns the guard off: drift is
                                     still measured and annotated on the pose,
                                     but never rejects it.
        clash_radius : float (1.8) - reject a pose with any heavy atom this
                                     close to a receptor atom. 0/None: off.
        max_affinity : float (0.0) - reject a pose whose minimizedAffinity is
                                     above this (positive = net repulsion, i.e.
                                     a clash the score itself reports).
                                     ``None`` turns it off.
        embed_timeout: float (60) - give up on one candidate's ConstrainedEmbed
                                     after this many seconds (see
                                     ``_run_constrained_embed`` -- RDKit's
                                     embedding has no native timeout and can,
                                     for a strained/flexible growth, run for a
                                     very long time or effectively never
                                     converge, blocking -- and in one observed
                                     production case, consuming unbounded
                                     memory in -- a whole concurrent batch).
    """

    def __init__(self, input_dict: dict):
        # Default the docking box to the fragment itself if no other site given.
        input_dict.setdefault("reference_path", input_dict.get("fragment_sdf"))
        super().__init__(input_dict)
        self.fragment_sdf = input_dict["fragment_sdf"]
        self.core, self._core_movable = _load_core(self.fragment_sdf,
                                                   input_dict.get("core_smarts"))
        guard = input_dict.get("max_core_rmsd", DEFAULT_MAX_CORE_RMSD)
        self.max_core_rmsd = None if guard is None else float(guard)
        radius = input_dict.get("clash_radius", DEFAULT_CLASH_RADIUS)
        self.clash_radius = float(radius) if radius else 0.0
        max_aff = input_dict.get("max_affinity", DEFAULT_MAX_AFFINITY)
        self.max_affinity = None if max_aff is None else float(max_aff)
        # Post-dock rejects, by reason -- reported in stats() so a run can say
        # how much of its library the pose guards threw away, and why.
        self.pose_rejections: Dict[str, int] = {}
        self.embed_timeout = float(input_dict.get("embed_timeout", _EMBED_TIMEOUT_DEFAULT))
        # The handle can turn about the bond into the core, so the product may be
        # buildable in a second orientation. Keep that template (and the receptor
        # to choose between them) only when there is actually something movable.
        flipped = _flipped_core(self.core, self._core_movable)
        self._alt_cores = [flipped] if flipped is not None else []
        # Loaded unconditionally: the clash guard measures every docked pose
        # against the receptor, not just the alternative core orientations.
        self._receptor_xyz = load_receptor_atoms(self.receptor_path)
        # Precompute reference core coordinates (receptor frame) for the guard.
        conf = self.core.GetConformer()
        self._core_ref_xyz = np.array(
            [list(conf.GetAtomPosition(i)) for i in range(self.core.GetNumAtoms())]
        )
        # Drift is measured on the atoms the pose really does fix: a movable atom
        # sits ~2 A away in the flipped orientation, which is a legitimate build,
        # not the broken binding mode the guard exists to catch.
        self._core_fixed = np.array([i for i in range(self.core.GetNumAtoms())
                                     if i not in set(self._core_movable)], dtype=int)

    # --- override hook 1: constrained 3D build --------------------------------
    def _prepare_pose(self, smiles: str) -> Tuple[Optional[str], Optional[str]]:
        return _constrained_pose_block(smiles, self.ph, self.core, self.seed,
                                       self.embed_timeout, self._alt_cores,
                                       self._receptor_xyz)

    # --- override hook 2: per-candidate box ------------------------------------
    def _box_flags(self, sdf_block: str) -> List[str]:
        """Size the box from *this candidate's own* just-built conformer
        (core pinned to the bound pose, the rest freely embedded by
        ``_prepare_pose``) instead of a box fixed once from the small
        original fragment. A route that's grown well past the fragment needs
        a box that covers the whole grown product, not one sized to the
        anchor alone -- otherwise gnina's local optimisation has to
        compromise the pose (including the anchored core) to fit, which the
        core-RMSD guard then has to reject as drift, when a correctly-sized
        box would have let the real elaboration dock cleanly."""
        mol = Chem.MolFromMolBlock(sdf_block, sanitize=False)
        if mol is None or mol.GetNumConformers() == 0:
            return super()._box_flags(sdf_block)  # shouldn't happen; same block just parsed fine to write it
        conf = mol.GetConformer()
        xyz = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
        lo, hi = xyz.min(axis=0), xyz.max(axis=0)
        center = (lo + hi) / 2
        # Pad each side by autobox_add (matching --autobox_add's own
        # per-side convention), floored at the class's configured/default
        # box size so a tiny early-route candidate doesn't get a starved box.
        size = np.maximum((hi - lo) + 2 * self.autobox_add, np.array(self.size, dtype=float))
        return [
            "--center_x", f"{center[0]:.3f}", "--center_y", f"{center[1]:.3f}", "--center_z", f"{center[2]:.3f}",
            "--size_x", f"{size[0]:.3f}", "--size_y", f"{size[1]:.3f}", "--size_z", f"{size[2]:.3f}",
        ]

    # --- override hook 3: which docked modes count -----------------------------
    def _pose_acceptable(self, pose: Chem.Mol) -> bool:
        """Reject a docked mode that broke the binding mode or is jammed into
        the receptor. Applied to every mode of the search, so a bad top-scored
        mode costs that mode, not the product.

        A rejected product ends up ``nan`` -- the same as a filtered one -- so
        Thompson Sampling is not rewarded for reaching it."""
        drift = self._core_drift(pose)
        if self.max_core_rmsd is not None and (drift is None or drift > self.max_core_rmsd):
            self._count_pose_rejection("core drift")
            return False
        if self.clash_radius and self._receptor_xyz is not None:
            if _receptor_clashes(pose, self._receptor_xyz, self.clash_radius):
                self._count_pose_rejection("clash")
                return False
        if self.max_affinity is not None:
            aff = self._parse_prop(pose, "minimizedAffinity")
            if aff is not None and np.isfinite(aff) and aff > self.max_affinity:
                self._count_pose_rejection("repulsive score")
                return False
        return True

    def _count_pose_rejection(self, reason: str) -> None:
        with self._lock:
            self.pose_rejections[reason] = self.pose_rejections.get(reason, 0) + 1

    def _progress_extra(self) -> str:
        # Called with the lock held (see GninaEvaluator._emit_progress), so read
        # the counter directly rather than through _count_pose_rejection's lock.
        if not self.pose_rejections:
            return ""
        return " | pose_rej " + str(dict(self.pose_rejections))

    def stats(self) -> dict:
        with self._lock:
            rejected = dict(self.pose_rejections)
        return {**super().stats(), "pose_rejections": rejected}

    # --- override the pose reader to annotate what the guards measured ---------
    def _best_pose(self, sdf_path: str, smiles: str):
        score, pose = super()._best_pose(sdf_path, smiles)
        if pose is None:
            return score, pose
        # Everything the guards judged the pose on, recorded on the pose: with a
        # guard switched off these are the only way to filter for it after the
        # run, and with it on they say how much margin the pose had.
        drift = self._core_drift(pose)
        if drift is not None:
            pose.SetProp("core_rmsd", f"{drift:.2f}")
        if self._receptor_xyz is not None:
            near = _receptor_contacts(pose, self._receptor_xyz)
            if near is not None:
                pose.SetProp("min_receptor_dist", f"{near:.2f}")
        return score, pose

    def _core_drift(self, pose: Chem.Mol) -> Optional[float]:
        """Heavy-atom RMSD of the conserved core in the docked pose vs the bound
        reference, in the receptor frame (no superposition -- absolute drift).
        Atoms whose position the bound pose does not fix (``_core_movable``,
        e.g. a carboxyl C=O whose -OH left) are excluded -- they are free to
        turn, so scoring them as drift would reject correctly anchored poses.

        When the core has graph symmetry (e.g. a symmetric ring/linker),
        ``GetSubstructMatch`` returns one arbitrary atom mapping, which can
        pair reference and pose atoms that aren't physically the same atom
        and so under-report real drift. Take the minimum RMSD over every
        symmetry-equivalent mapping instead -- the standard fix for RMSD
        under symmetry, and never an under-estimate of the true drift."""
        matches = pose.GetSubstructMatches(self.core, uniquify=False)
        matches = [m for m in matches if len(m) == self.core.GetNumAtoms()]
        if not matches:
            return None
        conf = pose.GetConformer()
        best = None
        keep = self._core_fixed
        if keep.size == 0:
            return 0.0
        for match in matches:
            xyz = np.array([list(conf.GetAtomPosition(i)) for i in match])
            d2 = ((xyz[keep] - self._core_ref_xyz[keep]) ** 2).sum(axis=1)
            rmsd = float(np.sqrt(d2.mean()))
            if best is None or rmsd < best:
                best = rmsd
        return best


# ---------------------------------------------------------------------------
# Standalone smoke-test:  python anchored_fragment_evaluator.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import sys
    if len(sys.argv) < 4:
        print("usage: anchored_fragment_evaluator.py receptor.pdb fragment.sdf "
              "'<core_smarts>' [product_smiles ...]")
        sys.exit(1)
    receptor, frag_sdf, core_smarts = sys.argv[1:4]
    products = sys.argv[4:] or []
    ev = AnchoredFragmentEvaluator({
        "receptor_path": receptor,
        "fragment_sdf": frag_sdf,
        "core_smarts": core_smarts,
        "cnn_scoring": "none",
    })
    for smi in products:
        s, reason = ev.evaluate(smi)
        print(f"{smi}\t{s}\t{reason}")
