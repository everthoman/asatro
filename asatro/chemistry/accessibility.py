"""Accessibility pre-pass: prune growth vectors that can only grow into the
protein.

Each reactive handle on the *bound* fragment implies a **growth vector** — the
direction the new substituent extends, taken from the bound pose as
(attachment atom -> the leaving group it displaces). Before the search spends any
docking budget we cast a cone along that vector and measure how far it reaches
before hitting receptor atoms. A vector that can't clear room for even a small
substituent is pruned; survivors are scored by how much open space they point
into.

This is the fast, geometry-only first cut (no docking). A later stub-growth pass
can refine the survivors. See DESIGN.md.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from rdkit import Chem

from asatro.chemistry.catalog import VOCAB
from asatro.chemistry.handles import analyze_fragment, neutralize, to_mol


@dataclass
class ProbeParams:
    step: float = 0.4            # march step along a direction (Å)
    max_reach: float = 6.0       # how far out we care to grow (Å)
    clash_radius: float = 2.6    # center-to-center below this = blocked by receptor (Å)
    cone_half_angle: float = 50.0  # cone aperture around the central vector (deg)
    n_cone: int = 40             # directions sampled within the cone
    open_depth: float = 3.0      # a direction counts as "open" if it reaches this (Å)
    min_free: float = 3.0        # vector accessible if its best direction reaches this (Å)


@dataclass
class ExitVector:
    fg_class: str
    attach_idx: int
    attach_pos: np.ndarray       # (3,)
    direction: np.ndarray        # (3,) unit vector
    leaving: tuple = ()          # fragment atom indices that leave (empty for amines)


# ---------------------------------------------------------------------------
# Receptor + fragment geometry
# ---------------------------------------------------------------------------
def load_receptor_atoms(pdb: str) -> np.ndarray:
    """Heavy-atom coordinates from a PDB (path or text). Skips hydrogens and
    waters; ignores connectivity (robust to messy protein PDBs)."""
    text = Path(pdb).read_text() if ("\n" not in pdb and Path(pdb).is_file()) else pdb
    pts: List[List[float]] = []
    for line in text.splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        if line[17:20].strip() in ("HOH", "WAT", "DOD"):  # skip waters
            continue
        # Element from cols 77-78 when present, else inferred from the atom name.
        element = (line[76:78].strip() or line[12:16].strip().lstrip("0123456789"))
        if element[:1].upper() == "H":  # skip hydrogens
            continue
        try:
            pts.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        except ValueError:
            continue
    return np.asarray(pts, dtype=float).reshape(-1, 3)


def _unit(v: np.ndarray) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else None


def growth_vectors(mol: Chem.Mol, fg_class: str) -> List[ExitVector]:
    """Exit vectors for every occurrence of ``fg_class`` on the *3D* fragment.

    For a handle with a leaving group the vector runs from the core attachment
    atom toward the leaving atom it's bonded to (where the new substituent lands).
    For an amine (nothing leaves) the vector is the N's open-valence direction
    (away from its heavy neighbours), since the new bond replaces an N-H.
    """
    if mol.GetNumConformers() == 0:
        return []
    conf = mol.GetConformer()
    pos = lambda i: np.array(conf.GetAtomPosition(i))
    fg_q = VOCAB.query[fg_class]
    lq = VOCAB.leaving.get(fg_class)
    vectors: List[ExitVector] = []

    for fg_match in mol.GetSubstructMatches(fg_q):
        fg_atoms = set(fg_match)
        if lq is None:
            # Amine: attach = the N (atom 0 of the class SMARTS); grow away from
            # the average of its heavy neighbours (the open valence).
            attach = fg_match[0]
            a = mol.GetAtomWithIdx(attach)
            nbr_dirs = [_unit(pos(n.GetIdx()) - pos(attach)) for n in a.GetNeighbors()]
            nbr_dirs = [d for d in nbr_dirs if d is not None]
            if not nbr_dirs:
                continue
            direction = _unit(-np.sum(nbr_dirs, axis=0))
            if direction is None:
                continue
            vectors.append(ExitVector(fg_class, attach, pos(attach), direction, ()))
            continue
        else:
            drop = set()
            for m in mol.GetSubstructMatches(lq):
                if any(x in fg_atoms for x in m):
                    drop.update(m)
            attach = lead = None
            for d in drop:
                for n in mol.GetAtomWithIdx(d).GetNeighbors():
                    if n.GetIdx() not in drop:
                        attach, lead = n.GetIdx(), d
                        break
                if attach is not None:
                    break
            if attach is None:
                continue
            direction = _unit(pos(lead) - pos(attach))
            if direction is None:
                continue
            vectors.append(ExitVector(fg_class, attach, pos(attach), direction,
                                      tuple(sorted(drop))))
    return vectors


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------
def _fibonacci_cone(axis: np.ndarray, half_angle_deg: float, n: int) -> np.ndarray:
    """~n unit directions within a cone of the given half-angle around ``axis``."""
    cos_lim = math.cos(math.radians(half_angle_deg))
    pts = []
    m = max(n * 12, 200)
    ga = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(m):
        z = 1.0 - (i + 0.5) * 2.0 / m
        r = math.sqrt(max(0.0, 1.0 - z * z))
        th = i * ga
        p = np.array([r * math.cos(th), r * math.sin(th), z])
        if float(np.dot(p, axis)) >= cos_lim:
            pts.append(p)
    dirs = [axis] + pts
    return np.array(dirs[: n + 1])


def _free_distance(p0: np.ndarray, d: np.ndarray, recat: np.ndarray, p: ProbeParams) -> float:
    """How far from ``p0`` along unit dir ``d`` before a receptor atom is within
    ``clash_radius`` (capped at ``max_reach``)."""
    if recat.size == 0:
        return p.max_reach
    t = p.step
    cr2 = p.clash_radius ** 2
    while t <= p.max_reach:
        pt = p0 + d * t
        if float(np.min(np.sum((recat - pt) ** 2, axis=1))) < cr2:
            return t
        t += p.step
    return p.max_reach


def probe_vector(ev: ExitVector, receptor: np.ndarray, p: ProbeParams = ProbeParams()) -> dict:
    """Free reach of an exit vector through a cone of growth directions."""
    # Crop the receptor to atoms that could possibly be hit — keeps it fast.
    if receptor.size:
        near = receptor[np.linalg.norm(receptor - ev.attach_pos, axis=1)
                        <= p.max_reach + p.clash_radius + 1.0]
    else:
        near = receptor
    dirs = _fibonacci_cone(ev.direction, p.cone_half_angle, p.n_cone)
    depths = np.array([_free_distance(ev.attach_pos, d, near, p) for d in dirs])
    free_central = float(depths[0])
    max_free = float(depths.max())
    return {
        "attach_idx": ev.attach_idx,
        "free_central": round(free_central, 2),
        "mean_free": round(float(depths.mean()), 2),
        "max_free": round(max_free, 2),
        "open_fraction": round(float(np.mean(depths >= p.open_depth)), 3),
        "accessible": max_free >= p.min_free,
    }


# ---------------------------------------------------------------------------
# Fragment-level assessment
# ---------------------------------------------------------------------------
def assess_fragment(mol: Chem.Mol, receptor: np.ndarray,
                    p: ProbeParams = ProbeParams()) -> dict:
    """Tier-1 analysis + accessibility pruning for a bound fragment in its pose.

    Adds an ``accessible`` flag and per-vector probe results to each compatible
    slot/reaction. A slot is accessible if any of its growth vectors is; a slot
    whose geometry can't be resolved is left accessible (we only prune when we're
    confident a direction is blocked)."""
    # Neutralize once (protonated amine / carboxylate -> neutral) and use this
    # single mol for both the analysis and the geometry probe, so atom indices
    # stay consistent. Hydrogens are implicit, so coordinates are preserved.
    mol = neutralize(mol)
    analysis = analyze_fragment(mol)
    for rid, info in analysis["reactions"].items():
        if not info["compatible"]:
            info["accessible"] = False
            continue
        rxn_ok = False
        for slot in info["slots"]:
            vecs = growth_vectors(mol, slot["fg_class"])
            probes = [probe_vector(v, receptor, p) for v in vecs]
            slot["vectors"] = probes
            slot["accessible"] = (not probes) or any(pr["accessible"] for pr in probes)
            rxn_ok = rxn_ok or slot["accessible"]
        info["accessible"] = rxn_ok
    analysis["accessible_reactions"] = sorted(
        rid for rid, info in analysis["reactions"].items() if info.get("accessible"))
    return analysis


def assess_from_files(fragment_sdf: str, receptor_pdb: str,
                      p: ProbeParams = ProbeParams()) -> dict:
    """Convenience: read the bound fragment SDF + receptor PDB and assess."""
    mol = Chem.MolFromMolFile(fragment_sdf, removeHs=True)
    if mol is None:
        raise ValueError(f"could not read fragment SDF: {fragment_sdf}")
    if mol.GetNumConformers() == 0:
        raise ValueError("fragment SDF has no 3D conformer (need the bound pose)")
    return assess_fragment(mol, load_receptor_atoms(receptor_pdb), p)
