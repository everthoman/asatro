"""Growth wiring: fragment-fixed route building + constrained placement.

These exercise everything up to (but not including) the gnina dock — product
enumeration with the bound fragment fixed, and the AnchoredFragmentEvaluator's
constrained pose generation. The dock itself needs the gnina binary + a GPU and
is not run here.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from asatro.chemistry.handles import derive_core
from asatro.engine.evaluators import MWEvaluator
from asatro.engine.route_sampler import RouteSampler
from asatro.growth import (build_growth_route, fragment_name_from_sdf,
                           fragment_smiles_from_sdf, make_evaluator,
                           resolve_fragment_name)


def _write_bound_fragment(tmp_path, smiles):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=7)
    AllChem.MMFFOptimizeMolecule(m)
    m = Chem.RemoveHs(m)
    p = tmp_path / "frag.sdf"
    Chem.MolToMolFile(m, str(p))
    return str(p)


def test_build_growth_route_places_fragment_and_library(tmp_path):
    bor = tmp_path / "boronic.smi"
    bor.write_text("OB(O)c1ccccc1 phB\nOB(O)c1ccncc1 pyB\n")
    files, route, summary = build_growth_route(
        ["suzuki"], "Brc1ccccc1", 1, [{0: str(bor)}], tmp_path)
    assert len(files) == 2 and files[1].endswith("fragment.smi")
    assert route[0][1:] == (2, None)
    # fragment file is a single fixed entry
    assert Chem.CanonSmiles(open(files[1]).read().split()[0]) == Chem.CanonSmiles("Brc1ccccc1")
    assert len(summary) == 1 and "bound fragment" in summary[0]


def test_build_growth_route_chains_an_extend_step(tmp_path):
    """Step 1 (start) fixes the fragment into a slot; step 2 (extend) is a real
    two-way reagent library on both its own component and (implicitly) the
    running intermediate -- mirrors combi's multi-step route shape, just with
    the fragment in step 1 instead of a real library."""
    bor = tmp_path / "boronic1.smi"
    bor.write_text("OB(O)c1ccccc1 phB\n")
    bor2 = tmp_path / "boronic2.smi"
    bor2.write_text("OB(O)c1ccncc1 pyB\n")
    files, route, summary = build_growth_route(
        ["suzuki", {"reaction_id": "suzuki", "slot": 1}], "Brc1ccc(Br)cc1", 1,
        [{0: str(bor)}, {0: str(bor2)}], tmp_path)
    assert len(files) == 3 and files[1].endswith("fragment.smi")
    assert [n for _smarts, n, _slot in route] == [2, 1]
    assert len(summary) == 2
    assert "Step 1" in summary[0] and "Step 2" in summary[1]


def test_build_growth_route_rejects_2component_later_step_with_no_slot(tmp_path):
    """A 2-component reaction reused for step 2+ needs an explicit slot
    naming which of its components binds the running intermediate -- any
    reaction can serve as an extend step now, not just the hand-authored
    role="extend" rows, so the old "must be an extend reaction" rule is gone;
    what's still required is knowing which slot the intermediate fills."""
    import pytest
    with pytest.raises(ValueError, match="give 'slot'"):
        build_growth_route(["suzuki", "suzuki"], "Brc1ccccc1", 1,
                           [{0: "a.smi"}, {0: "b.smi"}], tmp_path)


def test_build_growth_route_reuses_start_reaction_as_extend_step_with_slot(tmp_path):
    """A 2-component "start" reaction (no hand-authored extend counterpart
    needed) reused for step 2, with an explicit slot binding the
    intermediate -- the generalized extend path."""
    bor1 = tmp_path / "boronic1.smi"
    bor1.write_text("OB(O)c1ccccc1 phB\n")
    bor2 = tmp_path / "boronic2.smi"
    bor2.write_text("OB(O)c1ccncc1 pyB\n")
    files, route, summary = build_growth_route(
        ["suzuki", {"reaction_id": "suzuki", "slot": 1}], "Brc1ccc(Br)cc1", 1,
        [{0: str(bor1)}, {0: str(bor2)}], tmp_path)
    assert len(files) == 3 and files[1].endswith("fragment.smi")
    assert [n for _smarts, n, _slot in route] == [2, 1]
    assert route[1][2] == 1  # intermediate bound to slot 1 (the aryl-halide slot)
    assert len(summary) == 2


def test_route_sampler_grows_from_fragment(tmp_path):
    bor = tmp_path / "boronic.smi"
    bor.write_text("OB(O)c1ccccc1 phB\n")
    files, route, _summary = build_growth_route(
        ["suzuki"], "Brc1ccccc1", 1, [{0: str(bor)}], tmp_path)
    s = RouteSampler(mode="minimize")
    s.read_reagents(reagent_file_list=files, num_to_select=None)
    s.set_route(route)
    # component 0 has the boronic library; component 1 has the fixed fragment
    assert len(s.reagent_lists[0]) == 1 and len(s.reagent_lists[1]) == 1
    mol, smi, name, sel = s._build_product([0, 0])
    assert mol is not None
    assert Chem.MolToSmiles(mol) == Chem.CanonSmiles("c1ccc(-c2ccccc2)cc1")  # biphenyl


def test_route_sampler_grows_from_fragment_across_two_steps(tmp_path):
    """No docking, just enumeration: the fragment fills step 1's halide slot,
    step 1's boronic and step 2's boronic are both real (varying) libraries."""
    bor1 = tmp_path / "boronic1.smi"
    bor1.write_text("OB(O)c1ccccc1 phB\n")
    bor2 = tmp_path / "boronic2.smi"
    bor2.write_text("OB(O)c1ccncc1 pyB\n")
    files, route, _summary = build_growth_route(
        ["suzuki", {"reaction_id": "suzuki", "slot": 1}], "Brc1ccc(Br)cc1", 1,
        [{0: str(bor1)}, {0: str(bor2)}], tmp_path)
    s = RouteSampler(mode="minimize")
    s.read_reagents(reagent_file_list=files, num_to_select=None)
    s.set_route(route)
    assert [len(rl) for rl in s.reagent_lists] == [1, 1, 1]
    mol, smi, _name, _sel = s._build_product([0, 0, 0])
    assert mol is not None
    assert Chem.MolToSmiles(mol) == Chem.CanonSmiles("c1ccc(-c2ccc(-c3ccncc3)cc2)cc1")
    assert "Br" not in Chem.MolToSmiles(mol)  # both halide slots consumed


def test_route_sampler_rws_warmup_and_search(tmp_path):
    """The Roulette Wheel Selection path (warm_up_rws/search_rws), lifted from
    ts-gnina, is reachable from asatro's RouteSampler for a multi-reagent route
    -- exercised here with a cheap MW evaluator instead of a real dock."""
    bor = tmp_path / "boronic.smi"
    bor.write_text("\n".join(f"OB(O)c1ccc({'C' * i})cc1 phB{i}" for i in range(1, 7)) + "\n")
    files, route, _summary = build_growth_route(
        ["suzuki"], "Brc1ccccc1", 1, [{0: str(bor)}], tmp_path)
    s = RouteSampler(mode="maximize")
    s.set_hide_progress(True)
    s.read_reagents(reagent_file_list=files, num_to_select=None)
    s.set_route(route)
    s.set_evaluator(MWEvaluator())
    warmup = s.warm_up_rws(num_warmup_trials=2)
    assert warmup and all(len(row) == 3 for row in warmup)
    search = s.search_rws(num_targets=4, min_cpds_per_core=1, stop=100)
    assert isinstance(search, list)
    assert all(len(row) == 3 for row in search)


def test_anchored_evaluator_constrained_pose(tmp_path):
    sdf = _write_bound_fragment(tmp_path, "Brc1ccccc1")
    rec = tmp_path / "receptor.pdb"
    rec.write_text("ATOM      1  CA  ALA A   1      0.000   0.000   0.000  1.00  0.00           C\n")
    core = derive_core("Brc1ccccc1", "aryl_halide")  # benzene ring, halide excluded
    ev = make_evaluator(fragment_sdf=sdf, receptor_path=str(rec), core_smarts=core,
                        work_dir=str(tmp_path / "dock"))
    # Grow a biphenyl product and constrained-place it onto the bound benzene.
    block, err = ev._prepare_pose("c1ccc(-c2ccccc2)cc1")
    assert err is None and block is not None
    placed = Chem.MolFromMolBlock(block)
    assert placed is not None and placed.GetNumConformers() == 1
    # the conserved core must be present in the placed product
    assert placed.HasSubstructMatch(Chem.MolFromSmiles(core))


def test_anchored_evaluator_protonates_a_basic_amine_on_the_grown_part(tmp_path):
    """Regression: ``_constrained_pose_block`` used to parse the raw (neutral)
    product SMILES straight into RDKit, skipping the OpenBabel pH-protonation
    step the free (non-anchored) combi path always runs -- so a basic
    aliphatic amine picked up from a grown building block (piperidine,
    pyrrolidine, a primary amine, ...) came out of the constrained embed
    still neutral, docked and reported as the wrong ionization state. The
    fragment's own conserved core must stay neutral (it's excluded from
    protonation by construction, and matching is charge-insensitive anyway --
    see the comment in ``_constrained_pose_block``)."""
    sdf = _write_bound_fragment(tmp_path, "Brc1ccccc1")
    rec = tmp_path / "receptor.pdb"
    rec.write_text("ATOM      1  CA  ALA A   1      0.000   0.000   0.000  1.00  0.00           C\n")
    core = derive_core("Brc1ccccc1", "aryl_halide")  # benzene ring, halide excluded
    ev = make_evaluator(fragment_sdf=sdf, receptor_path=str(rec), core_smarts=core,
                        work_dir=str(tmp_path / "dock"))
    # Grow a biphenyl-piperidine product: the conserved benzene ring plus a
    # basic secondary amine (piperidine) on the newly-added ring.
    block, err = ev._prepare_pose("c1ccc(-c2ccc(C3CCNCC3)cc2)cc1")
    assert err is None and block is not None
    placed = Chem.MolFromMolBlock(block, removeHs=False)
    assert placed is not None and placed.GetNumConformers() == 1
    assert placed.HasSubstructMatch(Chem.MolFromSmiles(core))  # conserved core intact
    n_atoms = [a for a in placed.GetAtoms() if a.GetSymbol() == "N"]
    assert len(n_atoms) == 1
    n = n_atoms[0]
    assert n.GetFormalCharge() == 1                       # piperidine protonated ...
    assert n.GetTotalNumHs(includeNeighbors=True) == 2     # ... to a secondary ammonium


def test_anchored_evaluator_box_scales_with_the_actual_candidate(tmp_path):
    """Regression: the docking box used to be fixed once, sized only to the
    small original fragment (via a static --autobox_ligand reference) --
    every candidate in a route shared that one box no matter how far it had
    grown. Confirmed with a real gnina dock that a starved box lets
    --local_only's optimiser compromise the pose (including the anchored
    core) to fit, which the core-RMSD guard then has to reject as drift --
    a false negative caused by box sizing, not the elaboration itself. Now
    each candidate's own just-embedded conformer sizes its own box."""
    sdf = _write_bound_fragment(tmp_path, "Brc1ccccc1")
    rec = tmp_path / "receptor.pdb"
    rec.write_text("ATOM      1  CA  ALA A   1      0.000   0.000   0.000  1.00  0.00           C\n")
    core = derive_core("Brc1ccccc1", "aryl_halide")
    ev = make_evaluator(fragment_sdf=sdf, receptor_path=str(rec), core_smarts=core,
                        work_dir=str(tmp_path / "dock"))

    small_block, err = ev._prepare_pose("c1ccccc1")  # just the conserved core itself
    assert err is None
    small_flags = dict(zip(ev._box_flags(small_block)[0::2], ev._box_flags(small_block)[1::2]))

    # A long chain grown off the ring extends well past the fragment's own
    # tiny footprint -- the box must grow to cover it, not stay pinned to
    # the fragment's size/location.
    grown_block, err = ev._prepare_pose("c1ccc(CCCCCCCCCCCCCCCC)cc1")
    assert err is None
    grown_flags = dict(zip(ev._box_flags(grown_block)[0::2], ev._box_flags(grown_block)[1::2]))

    small_size = [float(small_flags[f"--size_{ax}"]) for ax in "xyz"]
    grown_size = [float(grown_flags[f"--size_{ax}"]) for ax in "xyz"]
    assert max(grown_size) > max(small_size) + 5  # materially bigger, not just noise

    # The small candidate's box floors out at the evaluator's configured
    # default size (nothing to grow into yet); the grown one exceeds it.
    assert small_size == list(ev.size)
    assert max(grown_size) > max(ev.size)


def test_anchored_evaluator_embed_timeout_kills_a_stuck_candidate(tmp_path):
    """Regression: AllChem.ConstrainedEmbed has no native timeout -- a real
    production job hung indefinitely (and grew to ~48GB RSS, one runaway
    candidate after another) when it got stuck, with no way to interrupt it
    and no way for cancel_event to help (it's only checked before a
    candidate starts, not while RDKit is mid-call). A first fix ran the embed
    in a worker *thread* and gave up waiting on timeout -- but that doesn't
    free anything, since Python threads can't be force-killed, so the
    abandoned computation (and its memory) kept running regardless. The
    fix must isolate the embed in a real, killable *process*.

    Exercised here with an absurdly small timeout (10ms -- shorter than
    process spawn itself takes) against an otherwise perfectly normal,
    fast-to-embed molecule: this can't rely on constructing a genuinely
    pathological molecule (slow/nondeterministic), but deterministically
    forces the same code path -- the worker is still starting up when the
    deadline passes, so _run_constrained_embed must terminate/kill it rather
    than block waiting, and _prepare_pose must return the failure quickly."""
    import time as _time

    sdf = _write_bound_fragment(tmp_path, "Brc1ccccc1")
    rec = tmp_path / "receptor.pdb"
    rec.write_text("ATOM      1  CA  ALA A   1      0.000   0.000   0.000  1.00  0.00           C\n")
    core = derive_core("Brc1ccccc1", "aryl_halide")
    ev = make_evaluator(fragment_sdf=sdf, receptor_path=str(rec), core_smarts=core,
                        work_dir=str(tmp_path / "dock"), embed_timeout=0.01)

    start = _time.monotonic()
    block, err = ev._prepare_pose("c1ccc(-c2ccccc2)cc1")
    elapsed = _time.monotonic() - start

    assert block is None
    assert "constrained embed failed" in err and "exceeded" in err
    assert elapsed < 10.0, f"took {elapsed:.2f}s -- kill+join should be fast, not block indefinitely"


def test_anchored_evaluator_max_core_rmsd_is_adjustable(tmp_path):
    sdf = _write_bound_fragment(tmp_path, "Brc1ccccc1")
    rec = tmp_path / "receptor.pdb"
    rec.write_text("ATOM      1  CA  ALA A   1      0.000   0.000   0.000  1.00  0.00           C\n")
    core = derive_core("Brc1ccccc1", "aryl_halide")
    ev = make_evaluator(fragment_sdf=sdf, receptor_path=str(rec), core_smarts=core,
                        work_dir=str(tmp_path / "dock"), max_core_rmsd=0.25)
    assert ev.max_core_rmsd == 0.25


def test_anchored_evaluator_organozinc_clean_removal(tmp_path):
    """negishi: organozinc's leaving_smarts (the whole ZnX group) removes cleanly
    -- a single, unambiguous case among the classes ported from ts-gnina."""
    sdf = _write_bound_fragment(tmp_path, "CC[Zn]Br")
    rec = tmp_path / "receptor.pdb"
    rec.write_text("ATOM      1  CA  ALA A   1      0.000   0.000   0.000  1.00  0.00           C\n")
    core = derive_core("CC[Zn]Br", "organozinc")
    assert core == "CC"  # Zn + Br both leave
    ev = make_evaluator(fragment_sdf=sdf, receptor_path=str(rec), core_smarts=core,
                        work_dir=str(tmp_path / "dock"))
    block, err = ev._prepare_pose("CCc1ccccc1")  # negishi product (ethylbenzene)
    assert err is None and block is not None
    placed = Chem.MolFromMolBlock(block)
    assert placed is not None and placed.HasSubstructMatch(Chem.MolFromSmiles(core))


def test_anchored_evaluator_alcohol_nothing_leaves(tmp_path):
    """williamson: alcohol's leaving_smarts is None (the O survives as an ether
    O, only its H is displaced) -- the "nothing leaves" default, same
    convention already used for amines."""
    sdf = _write_bound_fragment(tmp_path, "CCO")
    rec = tmp_path / "receptor.pdb"
    rec.write_text("ATOM      1  CA  ALA A   1      0.000   0.000   0.000  1.00  0.00           C\n")
    core = derive_core("CCO", "alcohol")
    assert core == "CCO"  # nothing leaves
    ev = make_evaluator(fragment_sdf=sdf, receptor_path=str(rec), core_smarts=core,
                        work_dir=str(tmp_path / "dock"))
    block, err = ev._prepare_pose("CCOCC")  # williamson product (diethyl ether)
    assert err is None and block is not None
    placed = Chem.MolFromMolBlock(block)
    assert placed is not None and placed.HasSubstructMatch(Chem.MolFromSmiles(core))


def test_fragment_smiles_from_sdf_roundtrip(tmp_path):
    sdf = _write_bound_fragment(tmp_path, "OC(=O)c1ccncc1")
    assert fragment_smiles_from_sdf(sdf) == Chem.CanonSmiles("OC(=O)c1ccncc1")


def _wall_receptor(path, points):
    """A PDB of pseudo-atoms at ``points`` — a wall for the anchored evaluator."""
    with open(path, "w") as fh:
        for i, p in enumerate(points, 1):
            fh.write(f"ATOM  {i:5d}  CA  ALA A{i:4d}    "
                     f"{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  1.00  0.00           C\n")
    return str(path)


def test_anchored_evaluator_builds_the_product_on_the_open_side(tmp_path):
    """Regression: the conserved core used to pin a carboxyl's C=O, so an amide
    could only be built where the -OH sat in the bound pose — into the wall,
    when that is where the -OH pointed. The C=O position is not conserved
    information (the whole group turns about the ring bond), so it is unpinned
    and the flipped core template is tried: the product must come out on the
    open side, clash-free."""
    sdf = _write_bound_fragment(tmp_path, "Cc1ccc(cc1)C(=O)O")
    frag = Chem.MolFromMolFile(sdf)
    conf = frag.GetConformer()
    c, o_carbonyl, o_h = frag.GetSubstructMatch(Chem.MolFromSmarts("[CX3](=O)[OX2H1]"))
    pos = lambda i: np.array(conf.GetAtomPosition(i))
    d = pos(o_h) - pos(c); d /= np.linalg.norm(d)
    u = np.cross(d, [0, 0, 1.0]); u /= np.linalg.norm(u)
    v = np.cross(d, u)
    centre = pos(o_h) + d * 2.8       # a wall 2.8 Å past the hydroxyl oxygen
    wall = [centre + a * u + b * v
            for a in np.arange(-9, 9.1, 1.2) for b in np.arange(-9, 9.1, 1.2)]
    rec = _wall_receptor(tmp_path / "receptor.pdb", wall)

    core = derive_core(frag, "carboxylic_acid")
    ev = make_evaluator(fragment_sdf=sdf, receptor_path=rec, core_smarts=core,
                        work_dir=str(tmp_path / "dock"))
    assert ev._core_movable and ev._alt_cores      # the C=O is free to turn

    block, err = ev._prepare_pose("Cc1ccc(cc1)C(=O)NC")   # the amide
    assert err is None and block is not None
    placed = Chem.MolFromMolBlock(block)
    n = [a.GetIdx() for a in placed.GetAtoms() if a.GetSymbol() == "N"]
    assert len(n) == 1
    n_pos = np.array(placed.GetConformer().GetAtomPosition(n[0]))
    # the amide N took the open C=O site, not the walled-in -OH site
    assert np.linalg.norm(n_pos - pos(o_carbonyl)) < np.linalg.norm(n_pos - pos(o_h))
    assert min(np.linalg.norm(np.array(w) - n_pos) for w in wall) > 3.0


# --- the fragment's own name ----------------------------------------------
# Product names are the reagent names joined by "_", so what the fragment slot
# is called is what makes a growth hit self-describing: TH17144_150266 says
# which fragment was grown; the old generic FRAG_150266 did not.

def _titled_sdf(tmp_path, title, smiles="Brc1ccccc1", name="frag.sdf"):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=7)
    m = Chem.RemoveHs(m)
    m.SetProp("_Name", title)
    p = tmp_path / name
    Chem.MolToMolFile(m, str(p))
    return str(p)


def test_fragment_name_comes_from_the_sdf_title(tmp_path):
    assert fragment_name_from_sdf(_titled_sdf(tmp_path, "TH17144")) == "TH17144"


def test_fragment_name_is_made_safe_for_a_reagent_file(tmp_path):
    """It has to sit in a .smi's whitespace-delimited name column, and then in
    a product name, so spaces and separators can't survive as-is."""
    assert fragment_name_from_sdf(
        _titled_sdf(tmp_path, "UNG2 hit 3 (batch/2)")) == "UNG2_hit_3_batch_2"
    assert fragment_name_from_sdf(_titled_sdf(tmp_path, "TH17144.sdf")) == "TH17144"
    long = fragment_name_from_sdf(_titled_sdf(tmp_path, "x" * 80))
    assert len(long) == 32


def test_a_prep_tools_temp_path_is_not_a_fragment_name(tmp_path):
    """What a bound fragment actually arrives with: OpenBabel writes its
    input's path into the molfile title, so every prepped fragment would
    otherwise be named after the temp file it was converted from."""
    for title in ("/tmp/tmp1kmnnxo3/ligand_raw.pdb",
                  "/tmp/tmpljlsafkv/ligand_protonated.pdb"):
        assert fragment_name_from_sdf(_titled_sdf(tmp_path, title)) == "FRAG"


def test_untitled_fragment_falls_back_to_the_filename_then_to_FRAG(tmp_path):
    assert fragment_name_from_sdf(
        _titled_sdf(tmp_path, "", name="TH17144.sdf")) == "TH17144"
    # ... but not to a filename that identifies nothing either
    assert fragment_name_from_sdf(_titled_sdf(tmp_path, "", name="frag.sdf")) == "FRAG"
    assert fragment_name_from_sdf(_titled_sdf(tmp_path, "   ")) == "FRAG"
    assert fragment_name_from_sdf(str(tmp_path / "missing.sdf")) == "FRAG"


def test_an_explicit_name_beats_whatever_the_file_says(tmp_path):
    """The field the user fills in is the reliable source; the SDF is only
    consulted when they leave it blank."""
    sdf = _titled_sdf(tmp_path, "/tmp/tmpX/ligand_raw.pdb")
    assert resolve_fragment_name("TH17144", sdf) == "TH17144"
    assert resolve_fragment_name("UNG2 hit 3", sdf) == "UNG2_hit_3"
    assert resolve_fragment_name("", sdf) == "FRAG"        # blank -> the file
    assert resolve_fragment_name(None, sdf) == "FRAG"
    assert resolve_fragment_name("  ", sdf) == "FRAG"


def test_growth_route_names_the_fragment_slot_after_the_sdf(tmp_path):
    """End to end: the one-entry reagent file the fragment fills carries the
    title, so every product built on it leads with that name."""
    boronic = tmp_path / "boronic.smi"
    boronic.write_text("OB(O)c1ccccc1\tBORON_1\n")
    files, _route, _summary = build_growth_route(
        ["suzuki"], "Brc1ccccc1", 1, [{0: str(boronic)}], tmp_path,
        fragment_name=fragment_name_from_sdf(_titled_sdf(tmp_path, "TH17144")))
    frag_file = next(f for f in files if f.endswith("fragment.smi"))
    assert open(frag_file).read().split() == ["Brc1ccccc1", "TH17144"]


def _displaced_pose_sdf(tmp_path, ev, product_smiles, shift, name="docked.sdf"):
    """A gnina-style output pose for ``product_smiles``: built anchored on the
    fragment, then rigidly translated ``shift`` A away, so its conserved core
    has drifted from the bound reference. The measured drift is smaller than
    ``shift`` itself: with a symmetric core the measure takes the best-matching
    mapping, so sliding the product along puts its far ring nearer the
    reference than the shift suggests -- what matters here is only that the
    drift lands far outside any sane guard."""
    block, err = ev._prepare_pose(product_smiles)
    assert err is None, err
    mol = Chem.MolFromMolBlock(block)
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (p.x + shift, p.y, p.z))
    mol.SetProp(ev.score_field, "-8.2")
    path = tmp_path / name
    w = Chem.SDWriter(str(path))
    w.write(mol)
    w.close()
    return str(path)


def _anchored_ev(tmp_path, **kw):
    sdf = _write_bound_fragment(tmp_path, "Brc1ccccc1")
    rec = tmp_path / "receptor.pdb"
    rec.write_text("ATOM      1  CA  ALA A   1      0.000   0.000   0.000  1.00  0.00           C\n")
    return make_evaluator(fragment_sdf=sdf, receptor_path=str(rec),
                          core_smarts=derive_core("Brc1ccccc1", "aryl_halide"),
                          work_dir=str(tmp_path / "dock"), **kw)


def test_anchored_evaluator_rejects_a_drifted_pose_when_the_guard_is_on(tmp_path):
    """Baseline for the switch below: a core sitting 5 A off its bound position
    is a broken binding mode, and the guard throws the pose away."""
    ev = _anchored_ev(tmp_path, max_core_rmsd=1.5)
    path = _displaced_pose_sdf(tmp_path, ev, "c1ccc(-c2ccccc2)cc1", 12.0)
    score, pose = ev._best_pose(path, Chem.CanonSmiles("c1ccc(-c2ccccc2)cc1"))
    assert (score, pose) == (None, None)


def test_anchored_evaluator_keeps_a_drifted_pose_when_the_guard_is_off(tmp_path):
    """max_core_rmsd=None switches the placement filter off: the same drifted
    pose is kept and scored, and the drift is still annotated on it so the run
    can be filtered on core_rmsd afterwards rather than during the search."""
    ev = _anchored_ev(tmp_path, max_core_rmsd=None)
    assert ev.max_core_rmsd is None
    smi = Chem.CanonSmiles("c1ccc(-c2ccccc2)cc1")
    path = _displaced_pose_sdf(tmp_path, ev, "c1ccc(-c2ccccc2)cc1", 12.0)
    score, pose = ev._best_pose(path, smi)
    assert pose is not None and score == -8.2
    assert float(pose.GetProp("core_rmsd")) > 5.0


# --- docking protocol + pose guards ----------------------------------------
# The anchored path used to pass --local_only: a local optimisation of the pose
# the constrained embed built, with no search. Measured on a real production
# run, that could not repair a starting pose built into the protein -- docked
# poses came back with a third of their atoms inside the receptor and
# minimizedAffinity of +10..+32, which a CNN score field ranks top regardless.
# Docking is a real search in the per-candidate box now, and the poses it
# returns are guarded on geometry, not only on score.

def _clashing_pose_sdf(tmp_path, smiles, receptor_xyz, score_field, value,
                       affinity=-8.0, name="docked.sdf"):
    """A docked pose sitting right on top of the receptor atoms."""
    from rdkit.Chem import AllChem
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=7)
    m = Chem.RemoveHs(m)
    conf = m.GetConformer()
    shift = np.array(receptor_xyz[0]) - np.array(list(conf.GetAtomPosition(0)))
    for i in range(m.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (p.x + shift[0], p.y + shift[1], p.z + shift[2]))
    m.SetProp(score_field, str(value))
    m.SetProp("minimizedAffinity", str(affinity))
    path = tmp_path / name
    w = Chem.SDWriter(str(path)); w.write(m); w.close()
    return str(path)


def _ev_with_receptor(tmp_path, **kw):
    """Anchored evaluator whose receptor has an atom at the fragment's own
    first core atom, so a pose left on the anchor is by definition clashing."""
    sdf = _write_bound_fragment(tmp_path, "Brc1ccccc1")
    frag = Chem.MolFromMolFile(sdf)
    p0 = frag.GetConformer().GetAtomPosition(1)
    rec = tmp_path / "receptor.pdb"
    rec.write_text("ATOM      1  CA  ALA A   1    %8.3f%8.3f%8.3f  1.00  0.00           C\n"
                   % (p0.x, p0.y, p0.z))
    return make_evaluator(fragment_sdf=sdf, receptor_path=str(rec),
                          core_smarts=derive_core("Brc1ccccc1", "aryl_halide"),
                          work_dir=str(tmp_path / "dock"), **kw)


def test_anchored_docking_is_a_real_search(tmp_path):
    """No --local_only: a local optimisation of the constrained embed cannot
    undo a grown arm the (receptor-blind) embed built into the protein, so the
    dock is a search, boxed to the candidate. There is no flag to opt back
    out -- the fast protocol was removed, not made optional."""
    ev = _ev_with_receptor(tmp_path)
    assert ev._extra_flags() == []
    assert not hasattr(ev, "local_only")


def test_anchored_guard_defaults_leave_room_for_a_searched_pose(tmp_path):
    """A correctly anchored pose from a free search sits further off the
    reference core than one that was never allowed to move -- re-docking a real
    run put clean poses at 1.2-1.8 A -- so the default guard is 2.0, not the
    1.5 that suited the local-only protocol."""
    ev = _ev_with_receptor(tmp_path)
    assert (ev.max_core_rmsd, ev.max_affinity) == (2.0, 0.0)


def test_pose_with_a_repulsive_score_is_rejected(tmp_path):
    """minimizedAffinity above zero is net repulsion: whatever the pose is, it
    is not a binding one. Rejected even when the score field (a CNN one, which
    does not see that term) looks excellent."""
    ev = _ev_with_receptor(tmp_path, score_field="CNN_VS", max_core_rmsd=None)
    smi = Chem.CanonSmiles("c1ccc(-c2ccccc2)cc1")
    path = _displaced_pose_sdf(tmp_path, ev, "c1ccc(-c2ccccc2)cc1", 0.0)
    pose = next(m for m in Chem.SDMolSupplier(path))
    pose.SetProp("CNN_VS", "4.31"); pose.SetProp("minimizedAffinity", "32.51")
    w = Chem.SDWriter(path); w.write(pose); w.close()
    assert ev._best_pose(path, smi) == (None, None)
    assert ev.stats()["pose_rejections"] == {"repulsive score": 1}
    # Off -> the same pose is kept and scored.
    off = _ev_with_receptor(tmp_path, score_field="CNN_VS", max_core_rmsd=None,
                            max_affinity=None)
    assert off._best_pose(path, smi)[0] == 4.31


def test_receptor_distance_is_annotated_even_though_nothing_rejects_on_it(tmp_path):
    """The clash guard is gone -- a real search never tripped it (zero
    rejections over a 278-product run, nothing within 2.5 A of the receptor).
    What it measured is still written onto every pose, so a run can be filtered
    on it afterwards if a pocket ever does misbehave."""
    ev = _ev_with_receptor(tmp_path, max_core_rmsd=None)
    smi = Chem.CanonSmiles("c1ccc(-c2ccccc2)cc1")
    # A pose right on top of a receptor atom, but scoring as if it binds: only
    # the geometry was ever objectionable, and nothing measures that any more.
    # score_field is minimizedAffinity here, so the helper's affinity is the score.
    path = _clashing_pose_sdf(tmp_path, smi, ev._receptor_xyz, ev.score_field, -9.0,
                              affinity=-8.0)
    score, pose = ev._best_pose(path, smi)
    assert pose is not None and score == -8.0          # kept, not rejected
    assert float(pose.GetProp("min_receptor_dist")) < 1.8
    assert ev.stats()["pose_rejections"] == {}


def test_a_rejected_mode_does_not_cost_the_whole_product(tmp_path):
    """Growth asks for one mode, but the selection is still per mode: given
    several (num_modes raised deliberately), the top-scored one being unusable
    says nothing about the rest, and the best *acceptable* mode wins. Only a
    product whose every mode is rejected scores nan."""
    ev = _ev_with_receptor(tmp_path, score_field="CNN_VS", num_modes=9)
    smi = Chem.CanonSmiles("c1ccc(-c2ccccc2)cc1")
    good = next(m for m in Chem.SDMolSupplier(
        _displaced_pose_sdf(tmp_path, ev, "c1ccc(-c2ccccc2)cc1", 0.0, name="a.sdf")))
    drifted = next(m for m in Chem.SDMolSupplier(
        _displaced_pose_sdf(tmp_path, ev, "c1ccc(-c2ccccc2)cc1", 12.0, name="b.sdf")))
    drifted.SetProp("CNN_VS", "9.99")      # best-scoring mode, off its anchor
    good.SetProp("CNN_VS", "3.10")
    path = str(tmp_path / "modes.sdf")
    w = Chem.SDWriter(path); w.write(drifted); w.write(good); w.close()

    score, pose = ev._best_pose(path, smi)
    assert score == 3.10                                   # not the 9.99 escapee
    assert float(pose.GetProp("core_rmsd")) < 2.0
    assert ev.stats()["pose_rejections"] == {"core drift": 1}


def test_anchored_docking_asks_for_one_mode(tmp_path):
    """Only the best pose is ever kept (one per product in the pose cache), so
    gnina is asked for one. The cost is that the pose guards become
    all-or-nothing: a rejected pose is a rejected product, with no lower-ranked
    mode left to fall back on."""
    ev = _ev_with_receptor(tmp_path)
    assert ev.num_modes == 1
    # Still overridable for a deliberate multi-mode run.
    assert _ev_with_receptor(tmp_path, num_modes=9).num_modes == 9


def test_unanchored_docking_keeps_its_nine_modes(tmp_path):
    """Combi has no pose guards; its _best_pose picks the best of gnina's modes
    by the configured score field, which is not gnina's own ordering -- so the
    extra modes there do change which pose wins."""
    from asatro.combi import make_evaluator as combi_evaluator
    rec = tmp_path / "receptor.pdb"
    rec.write_text("ATOM      1  CA  ALA A   1      0.000   0.000   0.000  1.00  0.00           C\n")
    ev = combi_evaluator(receptor_path=str(rec), center=(0.0, 0.0, 0.0),
                         work_dir=str(tmp_path / "dock"))
    assert ev.num_modes == 9
