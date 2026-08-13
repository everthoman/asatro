"""Per-structure MW/logP shown in the results gallery."""
from asatro.app import _enrich_top_props, _top_items
from asatro.svg import mol_props

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def test_mol_props_basic():
    p = mol_props(ASPIRIN)
    assert 179.0 <= p["mw"] <= 181.0     # aspirin ~180.16
    assert isinstance(p["logp"], float)
    assert mol_props("nonsense") == {"mw": None, "logp": None}


def test_top_items_carry_mw_and_logp():
    items = _top_items([(-7.5, ASPIRIN, "hit1")])
    assert items[0]["mw"] == mol_props(ASPIRIN)["mw"]
    assert items[0]["logp"] == mol_props(ASPIRIN)["logp"]
    assert items[0]["rank"] == 1 and items[0]["score"] == -7.5


def test_enrich_backfills_missing_props_from_smiles():
    # A pre-feature persisted result: top hit has no mw/logp.
    result = {"runs": [{"top": [{"score": -7.5, "smiles": ASPIRIN, "name": "h"}]}]}
    enriched = _enrich_top_props(result)
    hit = enriched["runs"][0]["top"][0]
    assert hit["mw"] == mol_props(ASPIRIN)["mw"]
    assert hit["logp"] == mol_props(ASPIRIN)["logp"]


def test_enrich_leaves_existing_props_untouched_and_tolerates_none():
    result = {"runs": [{"top": [{"smiles": ASPIRIN, "mw": 1.0, "logp": 2.0}]}]}
    _enrich_top_props(result)
    assert result["runs"][0]["top"][0]["mw"] == 1.0  # not recomputed
    assert _enrich_top_props(None) is None
    assert _enrich_top_props({}) == {}
