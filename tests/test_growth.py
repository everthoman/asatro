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
        ["suzuki"], "Brc1ccccc1", 0, [{1: str(bor)}], tmp_path)
    assert len(files) == 2 and files[0].endswith("fragment.smi")
    assert route == [("[c:1][Cl,Br,I].[c:2][B]([OX2])[OX2]>>[c:1][c:2]", 2)]
    # fragment file is a single fixed entry
    assert Chem.CanonSmiles(open(files[0]).read().split()[0]) == Chem.CanonSmiles("Brc1ccccc1")
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
        ["suzuki", "suzuki_ext_halide"], "Brc1ccc(Br)cc1", 0,
        [{1: str(bor)}, {0: str(bor2)}], tmp_path)
    assert len(files) == 3 and files[0].endswith("fragment.smi")
    assert [n for _smarts, n in route] == [2, 1]
    assert len(summary) == 2
    assert "Step 1" in summary[0] and "Step 2" in summary[1]


def test_build_growth_route_rejects_non_start_first_step(tmp_path):
    import pytest
    with pytest.raises(ValueError, match="must be a 'start'"):
        build_growth_route(["suzuki_ext_halide"], "Brc1ccccc1", 0, [{0: "dummy.smi"}], tmp_path)


def test_build_growth_route_rejects_non_extend_later_step(tmp_path):
    import pytest
    with pytest.raises(ValueError, match="must be an 'extend'"):
        build_growth_route(["suzuki", "suzuki"], "Brc1ccccc1", 0,
                           [{1: "a.smi"}, {1: "b.smi"}], tmp_path)


def test_route_sampler_grows_from_fragment(tmp_path):
    bor = tmp_path / "boronic.smi"
    bor.write_text("OB(O)c1ccccc1 phB\n")
    files, route, _summary = build_growth_route(
        ["suzuki"], "Brc1ccccc1", 0, [{1: str(bor)}], tmp_path)
    s = RouteSampler(mode="minimize")
    s.read_reagents(reagent_file_list=files, num_to_select=None)
    s.set_route(route)
    # component 0 has exactly the one fixed fragment; component 1 the boronic
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
        ["suzuki", "suzuki_ext_halide"], "Brc1ccc(Br)cc1", 0,
        [{1: str(bor1)}, {0: str(bor2)}], tmp_path)
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
        ["suzuki"], "Brc1ccccc1", 0, [{1: str(bor)}], tmp_path)
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
