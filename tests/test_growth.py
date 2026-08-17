"""Growth wiring: fragment-fixed route building + constrained placement.

These exercise everything up to (but not including) the gnina dock — product
enumeration with the bound fragment fixed, and the AnchoredFragmentEvaluator's
constrained pose generation. The dock itself needs the gnina binary + a GPU and
is not run here.
"""
from rdkit import Chem
from rdkit.Chem import AllChem

from asatro.chemistry.handles import derive_core
from asatro.engine.evaluators import MWEvaluator
from asatro.engine.route_sampler import RouteSampler
from asatro.growth import build_growth_route, make_evaluator, fragment_smiles_from_sdf


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
