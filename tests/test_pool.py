"""Master reagent pool: tagging, pruning, and the pool-backed resolver."""
from rdkit import Chem

from asatro.pool import Pool, pool_resolver, read_pool

POOL_TEXT = """\
CCC(=O)O acid1
NCc1ccccc1 amine1
O=Cc1ccccc1 aldehyde1
OB(O)c1ccccc1 boronic1
c1ccccc1 nohandle
CC(N)CC(=O)O.Cl aminoacid_salt
"""


def test_tagging_counts_and_untagged():
    p = Pool(read_pool(POOL_TEXT))
    c = p.counts()
    assert c.get("carboxylic_acid") == 2      # acid1 + the amino-acid
    assert c.get("primary_amine") == 2        # amine1 + the amino-acid
    assert c.get("aldehyde") == 1 and c.get("boronic") == 1
    assert "nohandle" not in [n for v in p.by_class.values() for _, n in v]


def test_salt_is_desalted_and_neutralized():
    p = Pool(read_pool(POOL_TEXT))
    acids = dict((n, s) for s, n in p.prune(["carboxylic_acid"]))
    # the HCl salt -> neutral, single fragment
    assert "." not in acids["aminoacid_salt"]
    assert "+" not in acids["aminoacid_salt"] and "-" not in acids["aminoacid_salt"]


def test_prune_unions_accepted_classes():
    p = Pool(read_pool(POOL_TEXT))
    names = [n for _, n in p.prune(["aldehyde", "ketone"])]
    assert names == ["aldehyde1"]
    # a difunctional block is reachable from either of its classes
    assert "aminoacid_salt" in [n for _, n in p.prune(["primary_amine"])]
    assert "aminoacid_salt" in [n for _, n in p.prune(["carboxylic_acid"])]


def test_pool_resolver_writes_pruned_smi(tmp_path):
    p = Pool(read_pool(POOL_TEXT))
    resolve = pool_resolver(p, str(tmp_path))
    path = resolve("suzuki", 1, ["boronic"])
    assert path is not None
    lines = [l for l in open(path).read().splitlines() if l.strip()]
    assert len(lines) == 1 and "boronic1" in lines[0]
    # a class with no members -> None (nothing to grow with)
    assert resolve("snar", 0, ["activated_aryl_halide"]) is None


# --- bundled pools ---------------------------------------------------------

def test_bundled_pools_exist_and_are_selectable():
    """Every bundled pool resolves to a real, non-empty .smi; an unknown id (or
    none) falls back to the default pool the app ships with."""
    from asatro.app import BUNDLED_POOLS, DEFAULT_POOL_PATH, POOL_DIR, bundled_pool_path

    ids = [p["id"] for p in BUNDLED_POOLS]
    assert "klara_sep_25" in ids and len(ids) == len(set(ids))
    for spec in BUNDLED_POOLS:
        path = POOL_DIR / spec["file"]
        assert path.is_file() and path.stat().st_size > 0
        assert bundled_pool_path(spec["id"]) == str(path)
    assert bundled_pool_path("nope") == DEFAULT_POOL_PATH
    assert bundled_pool_path("") == DEFAULT_POOL_PATH


def test_klara_pool_is_carbon_only_and_id_named():
    """The KLARA pool carries KLARA_IDs as block names and no carbon-free
    entries (inorganics are dropped when the SDF is converted)."""
    from asatro.app import bundled_pool_path

    rows = read_pool(bundled_pool_path("klara_sep_25"))
    assert len(rows) > 10_000
    for smiles, name in rows:
        assert name.isdigit(), f"{name!r} is not a KLARA_ID"
    for smiles, _ in rows[:500]:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        assert any(a.GetSymbol() == "C" for a in mol.GetAtoms()), smiles


def test_pool_preview_accepts_a_bundled_pool_id():
    from starlette.testclient import TestClient

    from asatro.app import app

    with TestClient(app) as client:
        r = client.post("/pool-preview", data={"pool_id": "klara_sep_25"})
    assert r.status_code == 200
    body = r.json()
    assert body["n_total"] > 10_000 and body["n_tagged"] > 0
