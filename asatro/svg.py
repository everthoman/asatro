"""Inline SVG structure rendering for the web UI's results gallery.

Structures are drawn once, when a job finishes, and stored in its
``results.json`` -- but the page's theme is a client-side toggle, so a single
stored SVG has to be readable in both. Every colour RDKit writes is therefore
swapped for a CSS variable (with the light value as its fallback), and
``palette_css`` emits the two sets of definitions. Switching theme then
recolours every structure on the page with no re-render and no round-trip.
"""
from __future__ import annotations

import functools
import re
from typing import Dict

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D

# One molecule carrying every element RDKit gives its own colour, used to read
# the palettes back out of the drawer.
_PALETTE_PROBE = "ICP(=O)(O)N(S)C(F)(Cl)Br"
_COLOUR = re.compile(r"#[0-9A-Fa-f]{6}")
_BACKGROUND_RECT = re.compile(r"<rect[^>]*?fill:#[0-9A-Fa-f]{6}[^>]*?>\s*(?:</rect>)?\s*")

# Element colours tuned per theme, taken from the in-house GNINA webapp
# (/opt/webapps/gnina), which measured them: RDKit's stock palette assumes a
# white background and its dark mode only lightens part of it, leaving several
# elements well under 4.5:1 against the card -- nitrogen at ~2.1:1 on the dark
# card, fluorine ~2.0:1 and sulfur ~1.7:1 on the light one. These keep the
# conventional element hues and move only the lightness. Elements not listed
# already clear 4.5:1 on both grounds.
_DARK_ATOM_PALETTE = {
    7:  (0.50, 0.65, 1.00),   # N -- light blue
    8:  (1.00, 0.48, 0.45),   # O -- salmon
    15: (1.00, 0.64, 0.30),   # P -- light orange
    35: (0.88, 0.64, 0.35),   # Br -- sand (RDKit's brown is near-black here)
    53: (0.75, 0.52, 0.99),   # I -- light violet
}
_LIGHT_ATOM_PALETTE = {
    8:  (0.80, 0.00, 0.00),   # O -- deeper red
    9:  (0.04, 0.44, 0.44),   # F -- teal
    15: (0.64, 0.28, 0.00),   # P -- burnt orange
    16: (0.43, 0.43, 0.00),   # S -- olive
    17: (0.00, 0.48, 0.00),   # Cl -- deeper green
}


def _draw(mol: Chem.Mol, width: int, height: int, dark: bool = False,
          stock: bool = False) -> str:
    """One depiction. ``stock`` draws RDKit's untuned light palette, which is
    only used to recognise the colours in SVGs stored before the tuning."""
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    if dark:
        rdMolDraw2D.SetDarkMode(drawer)
    if not stock:
        drawer.drawOptions().updateAtomPalette(
            _DARK_ATOM_PALETTE if dark else _LIGHT_ATOM_PALETTE)
    drawer.drawOptions().padding = 0.08
    # No background rect: the card behind it supplies the colour, in whichever
    # theme is showing. A painted one would be a white (or black) box sitting
    # on a card of the other colour.
    drawer.drawOptions().clearBackground = False
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


@functools.lru_cache(maxsize=1)
def _dark_colours() -> Dict[str, str]:
    """``light hex -> dark hex``, both taken from RDKit's own palettes --
    ``SetDarkMode`` is what keeps every atom legible on a dark ground (it
    lifts carbon/bonds to near-white and brightens N, O, Br and I), rather
    than a blanket filter over a light drawing.

    Read by rendering one probe molecule in each mode and pairing the colour
    literals in order: the two SVGs are structurally identical, so the n-th
    colour of one is the n-th of the other. Formatting ``getAtomPalette()``'s
    floats to hex here instead would be the obvious route and a silent trap --
    the drawer truncates where the obvious code rounds (0.802 -> CC, not CD),
    so the map would simply not match what it wrote."""
    mol = Chem.MolFromSmiles(_PALETTE_PROBE)
    dark = _COLOUR.findall(_draw(mol, 200, 160, dark=True))
    out: Dict[str, str] = {}
    # Stock alongside tuned, so a depiction stored before the tuning (or before
    # any theming at all) can still be recoloured from its literals alone --
    # both drawings are the same molecule, so their colours line up too.
    for light in (_draw(mol, 200, 160), _draw(mol, 200, 160, stock=True)):
        lights = _COLOUR.findall(light)
        if len(lights) != len(dark):   # palettes out of step: stay light-only
            return {}
        for lo, dk in zip(lights, dark):
            out.setdefault(lo.upper(), dk.upper())
    return out


def _var(light_hex: str) -> str:
    return f"--mol-{light_hex[1:].lower()}"


def _themed(svg: str) -> str:
    """Swap each colour literal for its CSS variable, keeping the light value
    as the fallback so the drawing still reads correctly anywhere the
    variables aren't defined (a copied-out SVG, an older page)."""
    colours = _dark_colours()

    def repl(m: re.Match) -> str:
        light = m.group(0).upper()
        return f"var({_var(light)},{m.group(0)})" if light in colours else m.group(0)

    return _COLOUR.sub(repl, svg)


def palette_css() -> str:
    """The ``--mol-*`` definitions for both themes, injected into the page.

    Dark sits on bare ``:root`` because dark is this UI's default theme; light
    overrides it under ``[data-theme="light"]``, matching how the rest of the
    stylesheet is written."""
    colours = _dark_colours()
    if not colours:
        return ""
    dark = " ".join(f"{_var(lo)}:{dk};" for lo, dk in sorted(colours.items()))
    light = " ".join(f"{_var(lo)}:{lo};" for lo in sorted(colours))
    # (Both sets include the stock-palette entries, so a re-themed old drawing
    #  resolves in either theme exactly as a freshly drawn one does.)
    return (f":root {{ {dark} }}\n"
            f'  :root[data-theme="light"] {{ {light} }}')


def retheme_svg(svg: str) -> str:
    """Bring a *stored* depiction up to the current theming, in place.

    A job's SVGs are drawn once and frozen into its ``results.json``, so a run
    that finished before the theming carries a painted white background and raw
    colour literals -- a white box on a dark card, for the life of the run.
    Re-drawing from the SMILES would fix it too, but this is the same drawing:
    strip the background rect and map the literals, no RDKit, no re-render, so
    it costs nothing to do on every read. Already-themed SVGs are returned
    untouched."""
    if not svg or "var(--mol-" in svg:
        return svg
    return _themed(_BACKGROUND_RECT.sub("", svg, count=1))


def mol_svg(smiles: str, width: int = 200, height: int = 160) -> str:
    """Render a SMILES to an inline, theme-aware SVG (XML declaration
    stripped), or "" if the SMILES doesn't parse."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    svg = _draw(mol, width, height)
    i = svg.find("<svg")
    return _themed(svg[i:] if i != -1 else svg)


def mol_props(smiles: str) -> dict:
    """Display descriptors for a gallery structure -- ``mw`` (g/mol) and cLogP
    -- using the same RDKit descriptors as the pre-dock MW/logP filters
    (``MolFilters``). Returns ``None`` values if the SMILES doesn't parse."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"mw": None, "logp": None}
    return {"mw": round(Descriptors.MolWt(mol), 1),
            "logp": round(Crippen.MolLogP(mol), 2)}
