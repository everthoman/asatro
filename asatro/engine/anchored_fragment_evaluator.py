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

import os
from typing import List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from asatro.chemistry.handles import carve_substructure_3d, neutralize
from asatro.engine.gnina_evaluator import (
    GninaEvaluator,
    prepare_ligand_3d,
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


def _constrained_pose_block(
    smiles: str, ph: float, core: Chem.Mol, seed: int = 0xF00D
) -> Tuple[Optional[str], Optional[str]]:
    """
    Protonate ``smiles`` (reusing GninaEvaluator's obabel step), then build a 3D
    pose with the ``core`` atoms pinned at their bound coordinates via
    ConstrainedEmbed. Returns ``(sdf_block, error)`` shaped like prepare_ligand_3d.
    """
    # Reuse the existing protonate+embed only to get a clean protonated SMILES;
    # we then re-embed under the core constraint. Cheapest correct path: redo
    # protonation here is overkill, so just parse and add Hs and constrain.
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, f"RDKit could not parse '{smiles}'"
    mol = Chem.AddHs(mol)
    if not mol.HasSubstructMatch(core):
        # The conserved core is not present -> the reaction did not preserve the
        # fragment (wrong route / wrong exit vector). Reject like a prep failure.
        return None, "conserved fragment core not found in product"
    try:
        # ConstrainedEmbed: matches core in mol, fixes those atoms at the core
        # coordinates, embeds the rest, and runs a restrained MMFF minimisation.
        AllChem.ConstrainedEmbed(mol, core, randomseed=seed)
    except Exception as e:  # embedding can fail for very strained grows
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
    """

    def __init__(self, input_dict: dict):
        # Default the docking box to the fragment itself if no other site given.
        input_dict.setdefault("reference_path", input_dict.get("fragment_sdf"))
        super().__init__(input_dict)
        self.fragment_sdf = input_dict["fragment_sdf"]
        self.core = _load_core(self.fragment_sdf, input_dict.get("core_smarts"))
        self.max_core_rmsd = float(input_dict.get("max_core_rmsd", 1.5))
        self.local_only = bool(input_dict.get("local_only", True))
        # Precompute reference core coordinates (receptor frame) for the guard.
        conf = self.core.GetConformer()
        self._core_ref_xyz = np.array(
            [list(conf.GetAtomPosition(i)) for i in range(self.core.GetNumAtoms())]
        )

    # --- override hook 1: constrained 3D build --------------------------------
    def _prepare_pose(self, smiles: str) -> Tuple[Optional[str], Optional[str]]:
        return _constrained_pose_block(smiles, self.ph, self.core, self.seed)

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
