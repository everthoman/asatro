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
2. Docking   -> add ``--local_only`` so gnina performs a local optimisation of
   the supplied pose instead of a global search; the core barely moves.

Plus a guard: after docking, the conserved-core atoms must not have drifted more
than ``max_core_rmsd`` A from the reference; if the grow broke the binding mode,
the product is rejected (``nan``) exactly like a filtered molecule.

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
from typing import List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

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


def _load_core(fragment_sdf: str, core_smarts: Optional[str]) -> Chem.Mol:
    """
    Build the conserved-core template (a 3D mol) used both to seed the embed and
    to check pose drift.

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
        return frag
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
        return carve_substructure_3d(frag, match)
    except ValueError as e:
        # More specific context than the generic carving error: here we know
        # it was core_smarts, not an arbitrary match, that cut the ring.
        raise ValueError(
            f"the conserved core carved from the fragment is not a valid "
            f"substructure ({e}). This usually means core_smarts excluded an "
            f"in-ring atom -- exclude only the reacting handle (the leaving "
            f"atom/group), and keep whole aromatic rings intact.") from e


_EMBED_TIMEOUT_DEFAULT = 60  # seconds

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


def _constrained_pose_block(
    smiles: str, ph: float, core: Chem.Mol, seed: int = 0xF00D,
    embed_timeout: float = _EMBED_TIMEOUT_DEFAULT,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Protonate ``smiles`` (reusing GninaEvaluator's obabel step), then build a 3D
    pose with the ``core`` atoms pinned at their bound coordinates via
    ConstrainedEmbed. Returns ``(sdf_block, error)`` shaped like prepare_ligand_3d.
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
    try:
        # ConstrainedEmbed: matches core in mol, fixes those atoms at the core
        # coordinates, embeds the rest, and runs a restrained MMFF minimisation.
        # Runs isolated in a child process (see _run_constrained_embed) -- the
        # embedded conformer comes back on `mol`, not mutated in place.
        mol = _run_constrained_embed(mol, core, seed, embed_timeout)
    except Exception as e:  # embedding can fail (or time out) for very strained grows
        return None, f"constrained embed failed: {e}"
    block = Chem.MolToMolBlock(mol) + "$$$$\n"
    lines = block.split("\n")
    if lines:
        lines[0] = "ligand"
    return _strip_sdf_properties("\n".join(lines)), None


class AnchoredFragmentEvaluator(GninaEvaluator):
    """
    GninaEvaluator that grows from a *bound* fragment: constrained embed onto the
    fragment pose + local-only gnina + a core-drift guard.

    Extra ``input_dict`` keys (on top of GninaEvaluator's)
        fragment_sdf : str        - fragment in its bound pose (3D SDF). REQUIRED.
                                     Also satisfies GninaEvaluator's "give me a
                                     site" requirement, though the box itself is
                                     no longer sized from it -- see _box_flags.
        core_smarts  : str        - the conserved sub-fragment (exclude the
                                     leaving handle). Strongly recommended.
        max_core_rmsd: float (1.5)- reject if core drifts more than this (A).
        local_only   : bool (True)- pass --local_only to gnina (no global search).
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
        self.core = _load_core(self.fragment_sdf, input_dict.get("core_smarts"))
        self.max_core_rmsd = float(input_dict.get("max_core_rmsd", 1.5))
        self.local_only = bool(input_dict.get("local_only", True))
        self.embed_timeout = float(input_dict.get("embed_timeout", _EMBED_TIMEOUT_DEFAULT))
        # Precompute reference core coordinates (receptor frame) for the guard.
        conf = self.core.GetConformer()
        self._core_ref_xyz = np.array(
            [list(conf.GetAtomPosition(i)) for i in range(self.core.GetNumAtoms())]
        )

    # --- override hook 1: constrained 3D build --------------------------------
    def _prepare_pose(self, smiles: str) -> Tuple[Optional[str], Optional[str]]:
        return _constrained_pose_block(smiles, self.ph, self.core, self.seed, self.embed_timeout)

    # --- override hook 2: docking flags ---------------------------------------
    def _extra_flags(self) -> List[str]:
        # --local_only: optimise the supplied (anchored) pose only; no global
        # search that would relocate the fragment. --minimize_iters keeps it short.
        return ["--local_only"] if self.local_only else []

    # --- override hook 3: per-candidate box ------------------------------------
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

    # --- override the pose reader to add the core-drift guard ------------------
    def _best_pose(self, sdf_path: str, smiles: str):
        score, pose = super()._best_pose(sdf_path, smiles)
        if pose is None:
            return score, pose
        drift = self._core_drift(pose)
        if drift is None or drift > self.max_core_rmsd:
            # The grow broke the binding mode: treat as a reject (nan score) so
            # TS does not reward it. Returning (None, None) makes _dock yield nan.
            return None, None
        pose.SetProp("core_rmsd", f"{drift:.2f}")
        return score, pose

    def _core_drift(self, pose: Chem.Mol) -> Optional[float]:
        """Heavy-atom RMSD of the conserved core in the docked pose vs the bound
        reference, in the receptor frame (no superposition -- absolute drift)."""
        match = pose.GetSubstructMatch(self.core)
        if not match or len(match) != self.core.GetNumAtoms():
            return None
        conf = pose.GetConformer()
        xyz = np.array([list(conf.GetAtomPosition(i)) for i in match])
        d2 = ((xyz - self._core_ref_xyz) ** 2).sum(axis=1)
        return float(np.sqrt(d2.mean()))


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
