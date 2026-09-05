# Brand assets

The flower mark is the primary logo for Underpainting. Everything here descends from
one Claude Design project:

<https://claude.ai/design/p/579191b1-6c12-4a28-8800-d7d6e993d292>

## Files

| File | What it is |
| --- | --- |
| `flower-logo-primary.png` | The mark, 2229×2100, transparent background. Use this one unless there's a reason not to. |
| `flower-logo-web.png` | The mark trimmed to the artwork and reduced to 260×247. This is the one the app actually loads, for the header and the favicon. The primary is 1.4MB, which is the wrong thing to send a phone for a 46px logo. |
| `flower-logo-conic.png` | Gradient variant — full spectrum, the same fill as the primary. |
| `flower-logo-linear.png` | Gradient variant — green/yellow, top-left to bottom-right. |
| `flower-logo-pastel.png` | Gradient variant — muted, lower saturation. |
| `flower-logo-radial.png` | Gradient variant — warm, orange/red from the centre out. |
| `flower-logo-v2-chalk.pdf` | The two-page sheet everything else was cut from. Page 2 also carries size tests on light and dark grounds, and a wordmark lockup. |

The mark is a five-petal flower with a heavy chalk-textured black outline, a gradient
fill, and a knocked-out white centre. The design sheet names it "primary mark — crayon
outline, conic spectrum."

## Two things to know before using these

**These are raster, not vector.** The PNGs were rendered at 6400px and cut down, so they
hold up at any size this project plausibly needs — favicon through print. But they are
pixels. Scaling past roughly 2200px wide will soften the chalk edge, and the colours
can't be re-tinted without re-rendering.

**The editable source isn't in this repo.** The real artboard is a `.dc.html` file
(plus a `support.js`) living in the Claude Design project linked above. Getting it out
needs a one-time `/design-login` authorisation that isn't set up on this machine, so
these assets were recovered from a PDF export instead. If the logo ever needs a genuine
change — different colours, a redrawn petal, an SVG for the web build — go back to the
Design project rather than editing a PNG.

## Why the PDF is tracked, when no other one is

`.gitignore` excludes `*.pdf` deliberately, and that rule carries a comment explaining
the reasoning: a planning PDF sitting in the repo root is one `git add .` away from a
public commit, and a broad pattern beats a filename list nobody remembers to update.

This folder is a scoped exception (`!assets/brand/*.pdf`) because brand source isn't
working notes, and because page 2 — the wordmark lockup and the on-dark size tests —
has no equivalent among the PNGs. The exception is one directory wide, so the original
rule's actual concern is still caught everywhere else.
