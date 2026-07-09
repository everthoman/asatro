"""Combi wiring: multi-step route building + plain (unanchored) evaluator.

Same scope boundary as test_growth.py: exercise everything up to (but not
including) the gnina dock itself — route/product enumeration and the plain
GninaEvaluator's free-embed pose prep. The dock needs the gnina binary + a GPU
and is not run here.
"""
import pytest
from rdkit import Chem

from asatro.chemistry.catalog import START_REACTIONS
from asatro.combi import build_combi_route, make_evaluator
from asatro.engine.evaluators import MWEvaluator
from asatro.engine.route_sampler import RouteSampler

# One hand-picked, class-matching example reagent per functional-group class --
# covers every start reaction's components (each accepts[0] is looked up here).
# Every entry validated to actually match its own class's detection SMARTS.
# Sourced from Hartenfeller_reactions.txt's own reference educts wherever a
# class is used as accepts[0] by some reaction; the 7 classes that only ever
# appear as a secondary accepts[] alternative (e.g. "secondary_amine" always
# trails "primary_amine") get a simple hand-picked example instead.
_CLASS_EXAMPLES = {
    "activated_aryl_halide": "c1cnc(F)cc1", "alcohol": "CCO", "aldehyde": "c1ccccc1C=O",
    "alkene": "c1ccccc1C=C", "alkyl_halide": "CCBr", "alpha_haloketone": "CC(=O)C(Br)C",
    "amidine": "N=C(N)NC", "aminophenol": "c1cc(O)c(N)cc1", "aminothiophenol": "c1c(S)c(N)ccc1",
    "anthranilic_acid": "c1c(C(=O)O)c(N)ccc1", "aryl_1_2_diketone": "c1ccccc1C(=O)C(=O)c1ccccc1",
    "aryl_halide": "c1ccccc1Br", "arylhydrazine": "c1ccccc1NN", "boronic": "c1ccccc1B(O)O",
    "carboxylic_acid": "CC(=O)O", "cyclic_ketone_6ring": "C1(=O)CCNCC1",
    "diketone_1_3": "CC(=O)CC(=O)C", "diketone_1_4": "CC(=O)CCC(=O)C", "formamide": "C(=O)N",
    "hydrazide": "NNC(=O)C", "hydrazine": "NNC", "imide_nh": "CC(=O)NC(=O)C",
    "indole_free_c3": "c1cccc2c1C=CN2", "internal_alkene": "c1ccccc1C(C)=CC",
    "internal_alkyne": "CC#CC", "isocyanate": "CN=C=O", "isothiocyanate": "CN=C=S",
    "ketone": "CCC(=O)C", "n_alkyl_diaminoarene": "c1c(NC)c(N)ccc1",
    "nheterocycle": "N1C=NC=C1", "nitrile": "CC#N",
    "ortho_activated_benzoic_acid": "c1c(C(=O)O)c([N+](=O)[O-])ccc1",
    "ortho_acylbenzoic_acid": "c1cc(C(=O)O)c(C(=O)C)cc1", "ortho_acylphenol": "c1cc(C(=O)C)c(O)cc1",
    "ortho_aminobenzaldehyde": "c1cccc(C=O)c1N", "ortho_halo_arylthioether": "c1cc(I)c(SC)cc1",
    "ortho_haloaniline": "c1cc(I)c(N)cc1", "ortho_halophenol": "c1cc(I)c(O)cc1",
    "phenethylamine": "c1cc(CCN)ccc1", "phenol": "c1ccccc1O", "primary_amine": "NC",
    "sulfonamide_nh": "CNS(=O)(=O)C", "sulfonyl_chloride": "CS(=O)(=O)Cl",
    "terminal_alkyne": "CC#C", "tetrazole_nh": "N1=NNC=N1", "thioamide": "NC(=S)C",
    "thiol": "CCS", "organozinc": "CC[Zn]Br", "acyl_halide": "CC(=O)Cl",
    "diaminoarene": "Nc1ccccc1N", "secondary_amine": "CCNCC",
    "phosphonate_ester": "CCOP(=O)(OCC)CC(=O)OCC", "azide": "CCN=[N+]=[N-]",
}


def _write(tmp_path, name, lines):
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def test_build_combi_route_single_start_step(tmp_path):
    boronic = _write(tmp_path, "boronic.smi", ["OB(O)c1ccccc1 phB", "OB(O)c1ccncc1 pyB"])
    halide = _write(tmp_path, "halide.smi", ["Brc1ccccc1 phBr"])
    files, route, summary = build_combi_route(["suzuki"], [[boronic, halide]], tmp_path)
    assert files == [boronic, halide]
    assert route[0][1:] == (2, None)
    assert len(summary) == 1 and "suzuki" not in summary[0]  # human name, not the id
    assert "Suzuki" in summary[0]


def test_build_combi_route_multi_step_start_then_extend(tmp_path):
    boronic1 = _write(tmp_path, "boronic1.smi", ["OB(O)c1ccccc1 phB"])
    dihalide = _write(tmp_path, "dihalide.smi", ["Brc1ccc(Br)cc1 dibromo"])
    boronic2 = _write(tmp_path, "boronic2.smi", ["OB(O)c1ccncc1 pyB"])
    files, route, summary = build_combi_route(
        ["suzuki", {"reaction_id": "suzuki", "slot": 1}],
        [[boronic1, dihalide], [boronic2]], tmp_path)
    assert files == [boronic1, dihalide, boronic2]
    assert [n for _smarts, n, _slot in route] == [2, 1]
    assert len(summary) == 2


def test_build_combi_route_reuses_start_reaction_as_extend_step_with_slot(tmp_path):
    """A 2-component "start" reaction (no hand-authored extend counterpart
    needed) reused for step 2, with an explicit slot binding the
    intermediate -- the generalized extend path."""
    boronic1 = _write(tmp_path, "boronic1.smi", ["OB(O)c1ccccc1 phB"])
    dihalide = _write(tmp_path, "dihalide.smi", ["Brc1ccc(Br)cc1 dibromo"])
    boronic2 = _write(tmp_path, "boronic2.smi", ["OB(O)c1ccncc1 pyB"])
    files, route, summary = build_combi_route(
        ["suzuki", {"reaction_id": "suzuki", "slot": 1}],
        [[boronic1, dihalide], [boronic2]], tmp_path)
    assert files == [boronic1, dihalide, boronic2]
    assert [n for _smarts, n, _slot in route] == [2, 1]
    assert route[1][2] == 1  # intermediate bound to slot 1 (the aryl-halide slot)
    assert len(summary) == 2


def test_build_combi_route_rejects_2component_later_step_with_no_slot(tmp_path):
    """Any reaction can serve as an extend step now, not just the
    hand-authored role="extend" rows -- but a 2-component reaction needs an
    explicit slot naming which component binds the running intermediate."""
    with pytest.raises(ValueError, match="give 'slot'"):
        build_combi_route(["suzuki", "suzuki"], [["a.smi", "b.smi"], ["c.smi", "d.smi"]], tmp_path)


def test_build_combi_route_rejects_reagent_count_mismatch(tmp_path):
    with pytest.raises(ValueError, match="needs 2 reagent file"):
        build_combi_route(["suzuki"], [["only_one.smi"]], tmp_path)


def test_build_combi_route_rejects_unknown_reaction(tmp_path):
    with pytest.raises(KeyError):
        build_combi_route(["not_a_reaction"], [["a.smi", "b.smi"]], tmp_path)


def test_build_combi_route_rejects_step_count_mismatch(tmp_path):
    with pytest.raises(ValueError, match="reagent_files has"):
        build_combi_route(
            ["suzuki", {"reaction_id": "suzuki", "slot": 1}], [["a.smi", "b.smi"]], tmp_path)


def test_route_sampler_builds_product_across_two_steps(tmp_path):
    """No fragment anywhere: both slots of the start step and the extend
    step's slot are real, independently varying reagent libraries."""
    boronic1 = _write(tmp_path, "boronic1.smi", ["OB(O)c1ccccc1 phB"])
    dihalide = _write(tmp_path, "dihalide.smi", ["Brc1ccc(Br)cc1 dibromo"])
    boronic2 = _write(tmp_path, "boronic2.smi", ["OB(O)c1ccncc1 pyB"])
    files, route, _summary = build_combi_route(
        ["suzuki", {"reaction_id": "suzuki", "slot": 1}],
        [[boronic1, dihalide], [boronic2]], tmp_path)

    s = RouteSampler(mode="minimize")
    s.read_reagents(reagent_file_list=files, num_to_select=None)
    s.set_route(route)
    assert [len(rl) for rl in s.reagent_lists] == [1, 1, 1]
    mol, smi, _name, _sel = s._build_product([0, 0, 0])
    assert mol is not None
    assert smi == Chem.CanonSmiles("c1ccc(-c2ccc(-c3ccncc3)cc2)cc1")
    assert "Br" not in Chem.MolToSmiles(mol)  # both halide slots consumed


def test_route_sampler_binds_intermediate_to_a_non_first_slot(tmp_path):
    """Reuse "schotten_baumann_amide" generically for step 2 with slot=0 --
    the intermediate must bind the *acid* pattern (RunReactants position 0,
    since this reaction's own component order is [acid, amine]), not
    position 1 -- proving _build_product's positional insertion actually
    respects intermediate_slot instead of always defaulting to position 0. A
    diacid step-1 reagent leaves one free -COOH on the intermediate for step
    2 to react through, confirmed directly against RDKit beforehand: the
    same intermediate mol fails to fire at position 1 (wrong pattern), only
    position 0 (the acid slot) works."""
    diacid = _write(tmp_path, "diacid.smi", ["OC(=O)c1ccc(C(=O)O)cc1 terephthalic"])
    amine1 = _write(tmp_path, "amine1.smi", ["CCN ethylamine"])
    amine2 = _write(tmp_path, "amine2.smi", ["CCCN propylamine"])
    files, route, _summary = build_combi_route(
        ["schotten_baumann_amide", {"reaction_id": "schotten_baumann_amide", "slot": 0}],
        [[diacid, amine1], [amine2]], tmp_path)
    assert route[1][2] == 0  # intermediate bound to the acid slot

    s = RouteSampler(mode="minimize")
    s.read_reagents(reagent_file_list=files, num_to_select=None)
    s.set_route(route)
    mol, smi, _name, _sel = s._build_product([0, 0, 0])
    assert mol is not None
    assert smi == Chem.CanonSmiles("CCNC(=O)c1ccc(C(=O)NCCC)cc1")


def test_route_sampler_searches_with_all_slots_variable(tmp_path):
    """Sanity check that a combi route with every slot variable (no fragment
    fixing any of them) drives warm-up + search the same way growth's RWS test
    exercises the fragment path, using a cheap MW evaluator instead of a real
    dock."""
    halide = _write(tmp_path, "halide.smi",
                    [f"Brc1ccc({'C' * i})cc1 br{i}" for i in range(1, 5)])
    boronic = _write(tmp_path, "boronic.smi",
                     [f"OB(O)c1ccc({'C' * i})cc1 b{i}" for i in range(1, 5)])
    files, route, _summary = build_combi_route(["suzuki"], [[boronic, halide]], tmp_path)

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


class _AlwaysFailEvaluator:
    """Every dock scores NaN, as if gnina couldn't place/dock anything for this
    fragment/receptor pairing -- regression coverage for warm_up() crashing
    (``np.min``/``np.max`` of an empty array) when nothing scores during
    warm-up instead of returning cleanly like warm_up_rws() already does."""
    def evaluate(self, mol):
        return float("nan")


def test_route_sampler_warm_up_returns_empty_when_every_dock_fails(tmp_path):
    halide = _write(tmp_path, "halide.smi", ["Brc1ccccc1 phBr", "Brc1ccc(C)cc1 tolBr"])
    boronic = _write(tmp_path, "boronic.smi", ["OB(O)c1ccccc1 phB", "OB(O)c1ccncc1 pyB"])
    files, route, _summary = build_combi_route(["suzuki"], [[boronic, halide]], tmp_path)

    s = RouteSampler(mode="maximize")
    s.set_hide_progress(True)
    s.read_reagents(reagent_file_list=files, num_to_select=None)
    s.set_route(route)
    s.set_evaluator(_AlwaysFailEvaluator())
    warmup = s.warm_up(num_warmup_trials=2)
    assert warmup == []


# A handful of reactions share a class with others but need a *stricter*
# member of it than the class-wide representative in _CLASS_EXAMPLES -- e.g.
# both "1H" and "2H" tetrazole tautomers satisfy the tetrazole_nh class (it
# matches either), but each Mitsunobu tetrazole *regioisomer* variant is
# written against one specific tautomer's atom positions (Hartenfeller's own
# reference reagents differ per variant for exactly this reason); the two
# nitro-position SNAr variants need an actual nitro group, not just any
# "activated_aryl_halide" class member (the class is deliberately broader,
# e.g. also covers pyridine-type activation). (component_index -> smiles)
_PER_REACTION_OVERRIDES = {
    "mitsunobu_tetrazole_3": {1: "N1N=NC=N1"},
    "mitsunobu_tetrazole_4": {1: "N1N=NC=N1"},
    "nucl_sub_aromatic_ortho_nitro": {0: "c1c([N+](=O)[O-])c(F)ccc1"},
    "nucl_sub_aromatic_para_nitro": {0: "c1c(F)ccc([N+](=O)[O-])c1"},
}


@pytest.mark.parametrize("rxn", START_REACTIONS, ids=[r["id"] for r in START_REACTIONS])
def test_every_start_reaction_builds_a_real_product(tmp_path, rxn):
    """Combi-path coverage of the full Hartenfeller-ported catalog: one
    hand-picked, class-matching reagent per component actually fires the
    reaction SMARTS and sanitizes -- catches SMARTS typos/bugs independent of
    the growth path's leaving_smarts (which this doesn't exercise at all)."""
    overrides = _PER_REACTION_OVERRIDES.get(rxn["id"], {})
    files = []
    for i, comp in enumerate(rxn["components"]):
        smi = overrides.get(i, _CLASS_EXAMPLES[comp["accepts"][0]])
        files.append(_write(tmp_path, f"r{i}.smi", [f"{smi} R{i}"]))
    s = RouteSampler(mode="minimize")
    s.set_hide_progress(True)
    s.read_reagents(reagent_file_list=files, num_to_select=None)
    s.set_route([(rxn["smarts"], len(rxn["components"]))])
    mol, smi, _name, _sel = s._build_product([0] * len(rxn["components"]))
    assert mol is not None, f"{rxn['id']} failed to build a product: {smi}"


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
