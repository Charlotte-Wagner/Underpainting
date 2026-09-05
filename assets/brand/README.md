# Brand assets

The flower mark is the primary logo for Underpainting. Everything here descends from
one Claude Design project:

<https://claude.ai/design/p/579191b1-6c12-4a28-8800-d7d6e993d292>

## Files

| File | What it is |
| --- | --- |
| `flower-logo-primary.png` | The mark, 2229×2100, transparent background. Use this one unless there's a reason not to. |
| `flower-logo-conic.png` | Gradient variant — full spectrum, the same fill as the primary. |
| `flower-logo-linear.png` | Gradient variant — green/yellow, top-left to bottom-right. |
| `flower-logo-pastel.png` | Gradient variant — muted, lower saturation. |
| `flower-logo-radial.png` | Gradient variant — warm, orange/red from the centre out. |
| `flower-logo-v2-chalk.pdf` | The two-page sheet everything else was cut from. Page 2 also carries size tests on light and dark grounds, and a wordmark lockup. **Untracked** — see below. |

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

## Why the PDF isn't tracked

`.gitignore` excludes `*.pdf` deliberately, and that rule carries a comment explaining
the reasoning: no PDF has ever belonged in this repository's tracked files, and a broad
pattern beats a filename list nobody remembers to update.

That rule still stands here, so `flower-logo-v2-chalk.pdf` sits in this folder locally
but does not go to GitHub. The PNGs beside it are tracked and carry the same artwork, so
nothing is lost to anyone cloning the repo — they just don't get the wordmark lockup and
the on-dark size tests, which live only on page 2 of the PDF.

If those are worth having in the repo, the fix is a narrow exception
(`!assets/brand/*.pdf`) rather than loosening the rule — but that's a deliberate change
to a deliberate policy, not a default.
