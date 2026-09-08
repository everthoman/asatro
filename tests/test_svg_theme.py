"""Theme-aware structure depictions.

A job's SVGs are drawn once and stored in results.json, but the page's theme
is a client-side toggle -- so one stored drawing has to read correctly in both.
"""
import re

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from starlette.testclient import TestClient

from asatro.app import app
from asatro.svg import (_LIGHT_ATOM_PALETTE, _dark_colours, mol_svg,
                        palette_css, retheme_svg)

SMILES = "Clc1ccc(cc1)C(=O)Nc1ccc(cc1)S(=O)(=O)N1CCOCC1"   # C, N, O, S, Cl
VAR = re.compile(r"var\((--mol-[0-9a-f]{6}),(#[0-9A-Fa-f]{6})\)")


def test_every_colour_becomes_a_variable_with_a_light_fallback():
    svg = mol_svg(SMILES)
    assert VAR.search(svg), "no themed colours emitted"
    # nothing left as a bare literal -- a stray one would stay black on a dark card
    assert not re.search(r"(?<![,(])#[0-9A-Fa-f]{6}", svg)


def test_resolving_the_variables_gives_back_the_light_drawing():
    """Every variable's fallback is the light-theme colour, so a drawing with
    the variables resolved is exactly the light depiction -- the theming adds
    the dark reading, it doesn't alter the light one."""
    drawer = rdMolDraw2D.MolDraw2DSVG(200, 160)
    drawer.drawOptions().updateAtomPalette(_LIGHT_ATOM_PALETTE)
    drawer.drawOptions().padding = 0.08
    drawer.drawOptions().clearBackground = False
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, Chem.MolFromSmiles(SMILES))
    drawer.FinishDrawing()
    plain = drawer.GetDrawingText()
    plain = plain[plain.find("<svg"):]
    assert VAR.sub(r"\2", mol_svg(SMILES)) == plain


def test_dark_values_come_from_rdkits_dark_mode_plus_the_contrast_tuning():
    """Not a filter over the light drawing: carbon and bonds lift to near-white
    and the elements RDKit's dark mode leaves too dark are moved further, which
    is what keeps every atom readable on the card."""
    colours = _dark_colours()
    assert colours["#000000"] == "#E5E5E5"      # carbon, bonds, symbols
    assert colours["#0000FF"] == "#7FA5FF"      # N: RDKit's dark blue is ~2.1:1
    assert colours["#7F4C19"] == "#E0A359"      # Br: RDKit's brown is near-black


def test_stock_rdkit_colours_are_recognised_too_for_stored_drawings():
    """Runs that finished before the tuning stored stock-palette literals; the
    map has to cover those or their atoms would keep the light colour."""
    colours = _dark_colours()
    assert colours["#FF0000"] == colours["#CC0000"]     # stock O and tuned O
    assert colours["#33CCCC"] == colours["#0A7070"]     # stock F and tuned F


def test_retheme_migrates_a_drawing_stored_before_the_theming():
    """A run finished under the old renderer has a painted white background and
    raw literals frozen in its results.json -- a white box on a dark card."""
    stored = ("<svg width='200px'>"
              "<rect style='opacity:1.0;fill:#FFFFFF;stroke:none' width='200.0' height='160.0'/>"
              "<path style='fill:none;stroke:#0000FF'/></svg>")
    out = retheme_svg(stored)
    assert "<rect" not in out                       # card colour shows through
    assert "var(--mol-0000ff,#0000FF)" in out
    assert retheme_svg(out) == out                  # idempotent
    assert retheme_svg(mol_svg(SMILES)) == mol_svg(SMILES)


def test_no_background_rect_so_the_card_shows_through():
    assert "<rect" not in mol_svg(SMILES)


def test_palette_css_defines_every_variable_a_drawing_uses_in_both_themes():
    css = palette_css()
    dark_block, light_block = css.split("\n")
    assert light_block.strip().startswith(':root[data-theme="light"]')
    for var, light_hex in set(VAR.findall(mol_svg(SMILES))):
        assert f"{var}:" in dark_block
        assert f"{var}:{light_hex}" in light_block   # light keeps RDKit's own value


def test_page_carries_the_palette():
    with TestClient(app) as client:
        html = client.get("/").text
    assert "__MOL_PALETTE__" not in html
    assert "--mol-000000:#E5E5E5" in html
    assert '--mol-000000:#000000' in html          # under the light override
