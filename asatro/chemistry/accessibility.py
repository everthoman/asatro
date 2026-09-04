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
class Rotamer:
    """One torsion state of a handle about the bond that joins it to the core.

    ``angle`` is the rotation applied to the bound pose (0 = as posed);
    ``free_xyz`` are the positions the handle's own movable atoms take in that
    state, so a rotamer that would ram them into the receptor can be rejected.
    """
    angle: float                 # degrees from the bound pose
    direction: np.ndarray        # (3,) unit exit vector in this state
    free_xyz: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))


@dataclass
class ExitVector:
    fg_class: str
    attach_idx: int
    attach_pos: np.ndarray       # (3,)
    direction: np.ndarray        # (3,) unit vector, as posed
    leaving: tuple = ()          # fragment atom indices that leave (empty for amines)
    free_atoms: tuple = ()       # kept atoms whose position the torsion moves (e.g. a
                                 # carboxyl's C=O when the -OH leaves)
    rotamers: tuple = ()         # Rotamer states, as-posed first


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


def rotate_about(points: np.ndarray, origin: np.ndarray, axis: np.ndarray,
                  angle_deg: float) -> np.ndarray:
    """Rodrigues rotation of ``points`` about the line (``origin``, unit ``axis``)."""
    th = math.radians(angle_deg)
    v = points - origin
    return (origin + v * math.cos(th)
            + np.cross(axis, v) * math.sin(th)
            + np.outer(v @ axis, axis) * (1 - math.cos(th)))


def torsion_axis(mol: Chem.Mol, attach: int, drop: set):
    """The bond the handle can spin about: ``(anchor, attach)`` where ``anchor``
    is the atom tying ``attach`` to the rest of the fragment, or ``None`` when
    the handle's direction is locked (an aromatic C-X: the bond into the ring is
    not rotatable, so the halide's direction is fixed by the ring geometry)."""
    a = mol.GetAtomWithIdx(attach)
    for nbr in a.GetNeighbors():
        i = nbr.GetIdx()
        if i in drop or nbr.GetAtomicNum() == 1:
            continue
        bond = mol.GetBondBetweenAtoms(attach, i)
        if bond.GetBondType() != Chem.BondType.SINGLE or bond.IsInRing():
            continue
        if nbr.GetDegree() < 2:   # terminal: spinning about it moves nothing
            continue
        return i
    return None


def _rotamer_angles(mol: Chem.Mol, attach: int) -> List[float]:
    """Torsion states to consider about the core-handle bond, in degrees.

    An sp2 attachment point (a carboxyl/acyl/aldehyde carbon) is planar and
    conjugated to what it hangs off: the only other minimum is the 180 deg flip,
    which swaps the leaving group with the group opposite it. An sp3 one (a
    sulfonyl S, a CH2-X) really does sweep its substituent around the bond, so
    walk the circle."""
    hyb = mol.GetAtomWithIdx(attach).GetHybridization()
    if hyb == Chem.HybridizationType.SP2:
        return [0.0, 180.0]
    return [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]


def growth_vectors(mol: Chem.Mol, fg_class: str) -> List[ExitVector]:
    """Exit vectors for every occurrence of ``fg_class`` on the *3D* fragment.

    For a handle with a leaving group the vector runs from the core attachment
    atom toward the leaving atom it's bonded to (where the new substituent lands).
    For an amine (nothing leaves) the vector is the N's open-valence direction
    (away from its heavy neighbours), since the new bond replaces an N-H.

    The bound pose pins one torsion state of the handle, which is *not* what the
    product is stuck with: each vector carries the ``rotamers`` its core bond can
    turn to (see ``_rotamer_angles``), as-posed first.
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
            # The bound pose fixes one torsion state of the handle, but the bond
            # into the core can spin: a carboxyl's -OH and its C=O swap on a 180
            # deg flip, so the substituent can leave along *either* C-O direction.
            # Enumerate those states, carrying the atoms that move with them.
            anchor = torsion_axis(mol, attach, drop)
            free = ()
            rotamers = [Rotamer(0.0, direction)]
            if anchor is not None:
                free = tuple(sorted(
                    n.GetIdx() for n in mol.GetAtomWithIdx(attach).GetNeighbors()
                    if n.GetIdx() not in drop and n.GetIdx() != anchor
                    and n.GetAtomicNum() != 1 and n.GetDegree() == 1))
                axis = _unit(pos(attach) - pos(anchor))
                if axis is not None:
                    free_xyz = np.array([pos(i) for i in free]).reshape(-1, 3)
                    rotamers = []
                    for ang in _rotamer_angles(mol, attach):
                        d = _unit(rotate_about(pos(lead).reshape(1, 3), pos(attach),
                                                axis, ang)[0] - pos(attach))
                        if d is None:
                            continue
                        rotamers.append(Rotamer(
                            ang, d, rotate_about(free_xyz, pos(attach), axis, ang)))
            vectors.append(ExitVector(fg_class, attach, pos(attach), direction,
                                      tuple(sorted(drop)), free, tuple(rotamers)))
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


def _rotamer_blocked(free_xyz: np.ndarray, near: np.ndarray, p: ProbeParams) -> bool:
    """Would turning to this rotamer push the handle's own atoms into the
    receptor? (An empty set of movable atoms never blocks.)"""
    if free_xyz.size == 0 or near.size == 0:
        return False
    d2 = ((near[None, :, :] - free_xyz[:, None, :]) ** 2).sum(axis=2)
    return bool((d2.min(axis=1) < p.clash_radius ** 2).any())


def probe_vector(ev: ExitVector, receptor: np.ndarray, p: ProbeParams = ProbeParams()) -> dict:
    """Free reach of an exit vector through a cone of growth directions.

    Every rotamer the handle can turn to is probed (a rotamer that would bury
    the handle's own atoms in the receptor is skipped); the result reports the
    best one, and ``rotamer`` says how far from the bound pose that state is.
    Growth is not committed to the torsion the pose happens to show, so pruning
    on the as-posed direction alone drops chemistry that is plainly reachable —
    e.g. a carboxylic acid whose -OH points at a wall while its C=O, 120 deg
    away and free to swap with it, points into open space."""
    # Crop the receptor to atoms that could possibly be hit — keeps it fast.
    if receptor.size:
        near = receptor[np.linalg.norm(receptor - ev.attach_pos, axis=1)
                        <= p.max_reach + p.clash_radius + 1.0]
    else:
        near = receptor
    rotamers = list(ev.rotamers) or [Rotamer(0.0, ev.direction)]

    best = None
    for rot in rotamers:
        if rot.angle and _rotamer_blocked(rot.free_xyz, near, p):
            continue
        dirs = _fibonacci_cone(rot.direction, p.cone_half_angle, p.n_cone)
        depths = np.array([_free_distance(ev.attach_pos, d, near, p) for d in dirs])
        cand = {
            "attach_idx": ev.attach_idx,
            "free_central": round(float(depths[0]), 2),
            "mean_free": round(float(depths.mean()), 2),
            "max_free": round(float(depths.max()), 2),
            "open_fraction": round(float(np.mean(depths >= p.open_depth)), 3),
            "accessible": float(depths.max()) >= p.min_free,
            "rotamer": rot.angle,
        }
        if best is None or cand["max_free"] > best["max_free"]:
            best = cand
    if best is None:   # every alternative rotamer was blocked; report as posed
        return probe_vector(ExitVector(ev.fg_class, ev.attach_idx, ev.attach_pos,
                                       ev.direction, ev.leaving), receptor, p)
    best["n_rotamers"] = len(rotamers)
    return best


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
