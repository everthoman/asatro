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
