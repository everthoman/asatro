"""Combi wiring: multi-step route building + plain (unanchored) evaluator.

Same scope boundary as test_growth.py: exercise everything up to (but not
including) the gnina dock itself — route/product enumeration and the plain
GninaEvaluator's free-embed pose prep. The dock needs the gnina binary + a GPU
and is not run here.
"""
import pytest
from rdkit import Chem

from asatro.combi import build_combi_route, make_evaluator
from asatro.engine.evaluators import MWEvaluator
from asatro.engine.route_sampler import RouteSampler


def _write(tmp_path, name, lines):
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def test_build_combi_route_single_start_step(tmp_path):
    halide = _write(tmp_path, "halide.smi", ["Brc1ccccc1 phBr"])
    boronic = _write(tmp_path, "boronic.smi", ["OB(O)c1ccccc1 phB", "OB(O)c1ccncc1 pyB"])
    files, route, summary = build_combi_route(["suzuki"], [[halide, boronic]], tmp_path)
    assert files == [halide, boronic]
    assert route == [("[c:1][Cl,Br,I].[c:2][B]([OX2])[OX2]>>[c:1][c:2]", 2)]
    assert len(summary) == 1 and "suzuki" not in summary[0]  # human name, not the id
    assert "Suzuki" in summary[0]


def test_build_combi_route_multi_step_start_then_extend(tmp_path):
    dihalide = _write(tmp_path, "dihalide.smi", ["Brc1ccc(Br)cc1 dibromo"])
    boronic1 = _write(tmp_path, "boronic1.smi", ["OB(O)c1ccccc1 phB"])
    boronic2 = _write(tmp_path, "boronic2.smi", ["OB(O)c1ccncc1 pyB"])
    files, route, summary = build_combi_route(
        ["suzuki", "suzuki_ext_halide"], [[dihalide, boronic1], [boronic2]], tmp_path)
    assert files == [dihalide, boronic1, boronic2]
    assert [n for _smarts, n in route] == [2, 1]
    assert len(summary) == 2


def test_build_combi_route_rejects_non_start_first_step(tmp_path):
    with pytest.raises(ValueError, match="must be a 'start'"):
        build_combi_route(["suzuki_ext_halide"], [["dummy.smi"]], tmp_path)


def test_build_combi_route_rejects_non_extend_later_step(tmp_path):
    with pytest.raises(ValueError, match="must be an 'extend'"):
        build_combi_route(["suzuki", "suzuki"], [["a.smi", "b.smi"], ["c.smi", "d.smi"]], tmp_path)


def test_build_combi_route_rejects_reagent_count_mismatch(tmp_path):
    with pytest.raises(ValueError, match="needs 2 reagent file"):
        build_combi_route(["suzuki"], [["only_one.smi"]], tmp_path)


def test_build_combi_route_rejects_unknown_reaction(tmp_path):
    with pytest.raises(KeyError):
        build_combi_route(["not_a_reaction"], [["a.smi", "b.smi"]], tmp_path)


def test_build_combi_route_rejects_step_count_mismatch(tmp_path):
    with pytest.raises(ValueError, match="reagent_files has"):
        build_combi_route(["suzuki", "suzuki_ext_halide"], [["a.smi", "b.smi"]], tmp_path)


def test_route_sampler_builds_product_across_two_steps(tmp_path):
    """No fragment anywhere: both slots of the start step and the extend
    step's slot are real, independently varying reagent libraries."""
    dihalide = _write(tmp_path, "dihalide.smi", ["Brc1ccc(Br)cc1 dibromo"])
    boronic1 = _write(tmp_path, "boronic1.smi", ["OB(O)c1ccccc1 phB"])
    boronic2 = _write(tmp_path, "boronic2.smi", ["OB(O)c1ccncc1 pyB"])
    files, route, _summary = build_combi_route(
        ["suzuki", "suzuki_ext_halide"], [[dihalide, boronic1], [boronic2]], tmp_path)

    s = RouteSampler(mode="minimize")
    s.read_reagents(reagent_file_list=files, num_to_select=None)
    s.set_route(route)
    assert [len(rl) for rl in s.reagent_lists] == [1, 1, 1]
    mol, smi, _name, _sel = s._build_product([0, 0, 0])
    assert mol is not None
    assert smi == Chem.CanonSmiles("c1ccc(-c2ccc(-c3ccncc3)cc2)cc1")
    assert "Br" not in Chem.MolToSmiles(mol)  # both halide slots consumed


def test_route_sampler_searches_with_all_slots_variable(tmp_path):
    """Sanity check that a combi route with every slot variable (no fragment
    fixing any of them) drives warm-up + search the same way growth's RWS test
    exercises the fragment path, using a cheap MW evaluator instead of a real
    dock."""
    halide = _write(tmp_path, "halide.smi",
                    [f"Brc1ccc({'C' * i})cc1 br{i}" for i in range(1, 5)])
    boronic = _write(tmp_path, "boronic.smi",
                     [f"OB(O)c1ccc({'C' * i})cc1 b{i}" for i in range(1, 5)])
    files, route, _summary = build_combi_route(["suzuki"], [[halide, boronic]], tmp_path)

    s = RouteSampler(mode="maximize")
    s.set_hide_progress(True)
    s.read_reagents(reagent_file_list=files, num_to_select=None)
    s.set_route(route)
    s.set_evaluator(MWEvaluator())
    warmup = s.warm_up(num_warmup_trials=2)
    assert warmup and all(len(row) == 3 for row in warmup)
    search = s.search(num_cycles=4)
    assert isinstance(search, list)
    assert all(len(row) == 3 for row in search)


def test_make_evaluator_reference_ligand_mode(tmp_path):
    rec = tmp_path / "receptor.pdb"
    rec.write_text("ATOM      1  CA  ALA A   1      0.000   0.000   0.000  1.00  0.00           C\n")
    ref = tmp_path / "ref.sdf"
    ref.write_text("mol\n\n\n  0  0  0  0  0  0  0  0  0  0999 V2000\nM  END\n$$$$\n")
    ev = make_evaluator(receptor_path=str(rec), reference_path=str(ref),
                        work_dir=str(tmp_path / "dock"))
    assert ev.reference_path == str(ref)
    assert ev.center is None
    # plain evaluator: no core pinning, no anchored-placement hooks
    assert ev._extra_flags() == []


def test_make_evaluator_center_mode(tmp_path):
    rec = tmp_path / "receptor.pdb"
    rec.write_text("ATOM      1  CA  ALA A   1      0.000   0.000   0.000  1.00  0.00           C\n")
    ev = make_evaluator(receptor_path=str(rec), center=(1.0, 2.0, 3.0), size=(20.0, 20.0, 20.0),
                        work_dir=str(tmp_path / "dock"))
    assert ev.reference_path is None
    assert ev.center == (1.0, 2.0, 3.0)
    assert ev.size == (20.0, 20.0, 20.0)


def test_make_evaluator_requires_binding_site(tmp_path):
    rec = tmp_path / "receptor.pdb"
    rec.write_text("ATOM      1  CA  ALA A   1      0.000   0.000   0.000  1.00  0.00           C\n")
    with pytest.raises(ValueError, match="reference_path or center"):
        make_evaluator(receptor_path=str(rec), work_dir=str(tmp_path / "dock"))


def test_prepare_pose_is_free_embed_not_constrained(tmp_path):
    """Contrast with test_growth's anchored pose test: the plain evaluator's
    _prepare_pose takes no core/fragment and just free-embeds the SMILES."""
    rec = tmp_path / "receptor.pdb"
    rec.write_text("ATOM      1  CA  ALA A   1      0.000   0.000   0.000  1.00  0.00           C\n")
    ev = make_evaluator(receptor_path=str(rec), center=(0.0, 0.0, 0.0),
                        work_dir=str(tmp_path / "dock"))
    block, err = ev._prepare_pose("c1ccc(-c2ccccc2)cc1")
    assert err is None and block is not None
    placed = Chem.MolFromMolBlock(block)
    assert placed is not None and placed.GetNumConformers() == 1
