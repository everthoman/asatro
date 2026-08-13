"""Downstream-reachability reagent pruning (asatro/chemistry/reachability.py).

A multi-step route should never enumerate a reagent whose intermediate can't
react at the next step. These check the exact prune, the safety guards (never
drop a viable reagent), and the clear error when a pool is emptied.
"""
import pytest

from asatro.chemistry.reachability import (
    UnreachableRouteError,
    prune_unreachable_reagents,
)

AMIDE = ("[C;$(C=O):1][OH1].[N;$(N[#6]);!$(N=*);!$([N-]);!$(N#*);!$([ND3]);"
         "!$([ND4]);!$(N[O,N]);!$(N[C,S]=[S,O,N]):2]>>[C:1][N+0:2]")
SUZUKI = ("[#6;H0;D3;$([#6](~[#6])~[#6]):1]B(O)O."
          "[#6;H0;D3;$([#6](~[#6])~[#6]):2][Cl,Br,I]>>[#6:2][#6:1]")


def _write(tmp_path, name, rows):
    p = tmp_path / name
    p.write_text("".join(f"{smi} {nm}\n" for smi, nm in rows))
    return str(p)


def _names(smi_path):
    return {line.split()[1] for line in open(smi_path) if line.strip()}


def test_prunes_acids_that_cannot_complete_the_suzuki(tmp_path):
    # Route: amide (fragment fills the amine slot) -> suzuki with the
    # intermediate bound to the aryl-halide slot (slot 1), boronic as the fresh
    # reagent. So an acid survives only if its amide intermediate carries an
    # aryl halide for the Suzuki to fire on.
    acids = _write(tmp_path, "acids.smi", [
        ("OC(=O)c1ccc(Br)cc1", "brbenzoic"),   # aryl halide -> keep
        ("OC(=O)c1ccccc1Cl", "clbenzoic"),     # aryl halide -> keep
        ("CC(=O)O", "acetic"),                 # no handle   -> drop
        ("OCC(=O)O", "glycolic"),              # no handle   -> drop
    ])
    frag = _write(tmp_path, "frag.smi", [("NCc1ccccc1", "FRAG")])
    boronic = _write(tmp_path, "boronic.smi",
                     [("OB(O)c1ccccc1", "phB"), ("OB(O)c1ccncc1", "pyB")])
    route = [(AMIDE, 2, None), (SUZUKI, 1, 1)]
    files = [acids, frag, boronic]

    out = prune_unreachable_reagents(route, files, work_dir=str(tmp_path / "r"))

    assert _names(out[0]) == {"brbenzoic", "clbenzoic"}  # acid pool narrowed
    assert out[1] == frag       # the fixed 1-entry fragment file is untouched
    assert out[2] == boronic    # the last step is never pruned


def test_empty_pool_raises_a_clear_error(tmp_path):
    acids = _write(tmp_path, "acids.smi",
                   [("CC(=O)O", "acetic"), ("OCC(=O)O", "glycolic")])  # none viable
    frag = _write(tmp_path, "frag.smi", [("NCc1ccccc1", "FRAG")])
    boronic = _write(tmp_path, "boronic.smi", [("OB(O)c1ccccc1", "phB")])
    route = [(AMIDE, 2, None), (SUZUKI, 1, 1)]

    with pytest.raises(UnreachableRouteError):
        prune_unreachable_reagents(route, [acids, frag, boronic],
                                   work_dir=str(tmp_path / "r"))


def test_two_varying_pools_left_unpruned(tmp_path):
    # combi-style step 0: amide with BOTH acid and amine as real libraries (no
    # fixed fragment). Two varying pools -> the exact single-degree-of-freedom
    # prune doesn't apply, so nothing is dropped (safe: no false negatives).
    acids = _write(tmp_path, "acids.smi",
                   [("OC(=O)c1ccc(Br)cc1", "br"), ("CC(=O)O", "ac")])
    amines = _write(tmp_path, "amines.smi",
                    [("NCc1ccccc1", "bn"), ("NCCc1ccccc1", "pe")])
    boronic = _write(tmp_path, "boronic.smi", [("OB(O)c1ccccc1", "phB")])
    route = [(AMIDE, 2, None), (SUZUKI, 1, 1)]
    files = [acids, amines, boronic]

    out = prune_unreachable_reagents(route, files, work_dir=str(tmp_path / "r"))

    assert out == files  # unchanged


def test_single_step_route_is_a_noop(tmp_path):
    boronic = _write(tmp_path, "boronic.smi", [("OB(O)c1ccccc1", "phB")])
    halide = _write(tmp_path, "halide.smi", [("Brc1ccccc1", "phBr")])
    route = [(SUZUKI, 2, None)]
    files = [boronic, halide]

    assert prune_unreachable_reagents(route, files, work_dir=str(tmp_path / "r")) == files


def test_three_step_route_prunes_an_intermediate_step(tmp_path):
    # Exercises pruning of a non-first, non-last step (the k>0 path, where the
    # running intermediate binds a slot): amide -> suzuki -> amide. Step-1
    # boronics survive only if their Suzuki product can react in the final
    # amide, i.e. only the boronic that carries a carboxylic acid.
    acids = _write(tmp_path, "acids.smi",
                   [("OC(=O)c1ccc(Br)cc1", "brbz"), ("CC(=O)O", "ac")])
    frag = _write(tmp_path, "frag.smi", [("NCc1ccccc1", "FRAG")])
    boronics = _write(tmp_path, "boronics.smi", [
        ("OB(O)c1ccc(C(=O)O)cc1", "boronobenzoic"),  # carries COOH -> keep
        ("OB(O)c1ccccc1", "phenyl"),                 # no COOH      -> drop
    ])
    amines = _write(tmp_path, "amines.smi", [("NCc1ccccc1", "bn")])  # last step
    # amide -> suzuki(intermediate binds aryl-halide slot 1) -> amide(intermediate
    # binds acid slot 0)
    route = [(AMIDE, 2, None), (SUZUKI, 1, 1), (AMIDE, 1, 0)]
    files = [acids, frag, boronics, amines]

    out = prune_unreachable_reagents(route, files, work_dir=str(tmp_path / "r"))

    assert _names(out[0]) == {"brbz"}            # step 0 acid pool pruned
    assert _names(out[2]) == {"boronobenzoic"}   # step 1 boronic pool pruned
    assert out[3] == amines                      # final step untouched


def test_prune_reflects_the_wiring_choice(tmp_path):
    # Same acids, but wire the Suzuki the other way (intermediate bound to the
    # boronic slot 0, aryl halide as the fresh reagent). Now an acid survives
    # only if its intermediate carries a *boronic* handle -- the opposite
    # subset -- confirming the prune tracks the actual route wiring, not a
    # hard-coded handle.
    acids = _write(tmp_path, "acids.smi", [
        ("OC(=O)c1ccc(Br)cc1", "brbenzoic"),        # aryl halide, no boron -> drop
        ("OC(=O)c1ccc(B(O)O)cc1", "boronobenzoic"),  # boronic handle      -> keep
    ])
    frag = _write(tmp_path, "frag.smi", [("NCc1ccccc1", "FRAG")])
    halide = _write(tmp_path, "halide.smi",
                    [("Brc1ccccc1", "phBr"), ("Brc1ccncc1", "pyBr")])
    route = [(AMIDE, 2, None), (SUZUKI, 1, 0)]  # intermediate binds boronic slot 0
    files = [acids, frag, halide]

    out = prune_unreachable_reagents(route, files, work_dir=str(tmp_path / "r"))

    assert _names(out[0]) == {"boronobenzoic"}
