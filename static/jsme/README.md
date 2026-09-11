# JSME (vendored)

The JSME molecule editor, used by the "Draw" button behind the growth run's
**Measure drift on** field (`templates/index.html`). Vendored rather than loaded
from a CDN: this box's apps run without outbound internet, and an editor that
silently fails to load is worse than a plain text field.

JSME is by Peter Ertl and Bruno Bienfait — <https://jsme-editor.github.io/> —
free for academic and commercial use. GWT-compiled output, copied as-is from the
sibling app at `/opt/webapps/gnina/static/jsme`; nothing here is edited by hand.
`jsme.nocache.js` is the entry point and loads the `*.cache.js` permutations it
needs.
