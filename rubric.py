"""Teaching rubric for Underpainting's step-by-step writeup, and prompt assembly.

WHERE THIS CAME FROM
---------------------
Written unaided, from two years of studio art, before any model saw it. The unedited v1 is
kept at teaching-rubric-draft.md (gitignored). Reviewed against the build plan's three ⬥5
critique questions: where is this generic, where would an experienced painter disagree,
what's missing that a beginner specifically needs.

The critique was taken and the edits were then applied here rather than by hand, which is a
deliberate departure from ⬥5's "take the critique, not the prose." The ideas, the sequence,
the beginner mistakes and the emphasis argument are all from v1. Full record of what changed
and why is in debug-log.md under ⬥5.

Three changes are worth flagging here because they alter the method rather than the wording:

  - v1 said darks first, then midtones, then brightest last. The value study image contains
    true black and true white from what was then stage 1, so the text and the image disagreed
    about what step 1 is. Resolved toward the image: both endpoints get placed first as
    anchors, with the brightest accents still saved for last, since a lightest mass and a
    final highlight are two different things.
  - v1 said "color isn't what gives a piece form, values do that," while the app measures
    temperature as one of only two stats it sends. Softened so the two stop contradicting
    each other.
  - v1 stated the warm-light/cool-distance rule as absolute. Measured on the primary test
    photo, all six palette swatches sit on the cool side of neutral, so an absolute reading
    would have a beginner warming the foreground incorrectly. Now framed as relative and as
    a default to test.

Added because a beginner needs them and would not know to ask: toning the ground (the app is
called Underpainting and v1 never mentioned it), hard versus soft edges, what squinting
actually does, how the suggested tubes connect to the stages, and a stopping criterion.

Structured in build order on purpose: the drawing, then darks and lights, then midtones, then
full detail, because that's the same order imaging.build_stages computes its four outputs in.
The rubric isn't describing painting in general, it's describing how to read the four
specific images this app generates for a specific photo, in the order they're generated.

THE CONTENT AND VOICE HERE ARE NOT UP FOR A MODEL REWRITE
-----------------------------------------------------------
Editing this text is a by-hand job, not something a coding session does. See ⬥5 in
debug-log.md: the moment the rubric is in the model's voice instead of the painter's, the
one part of this project that's actually theirs is gone.

One exception, recorded rather than left to be discovered. The S12 stage redesign renamed the
four stages, which left this text describing images that no longer exist. The nineteen
paragraphs were re-filed under the new headings by a session, mechanically and byte for byte,
which is a move rather than a rewrite. Six sentences were then patched by that session, at
Charlotte's explicit request and on the reasoning that she wanted the pipeline working end to
end before tuning wording: the two that had become factually false (the value study "from the
very first stage", and "the value structure before color" in the Throughline), the clause
saying value comes first when the drawing now does, the S9 note above, and one added sentence
tying the toned ground to the gray visible in stages 1 and 2. A seventh change, also hers to
review: the "This stage confirms the block-in holds up" paragraph, written about soft focus
and redundant with the squint-and-check paragraph in Full detail, was cut rather than patched,
on her instruction after she read it flagged. Eighteen paragraphs remain, all of them
untouched. The patched sentences and the one cut are pending her pass and are the only content
in this file a session has touched.
"""

from imaging import STAGE_CAPTIONS

RUBRIC = """\
Drawing

Tone the canvas before anything else. A thin wash of a mid-value neutral, wiped over the
whole surface and left to dry, replaces the white ground with something in the middle of the
value range. This is what makes the rest of the order work: on a mid-toned ground you build
in both directions, darks down and lights up, and each reads correctly as you place it. On
raw white, everything looks dark by comparison and the whole value scale gets compressed
before any real judgment is possible. In the first two stage images that ground is the flat
gray: it is canvas you have not covered yet, not a color you mixed.

Then get the drawing and the proportions down, loosely, before any value work starts. No
amount of correct value work fixes a wrong drawing underneath it.

Darks and lights

Value carries most of the description of form, which is why it comes before color, but color
is not only decoration: temperature does real work in pushing things back and pulling them
forward, and that becomes a live concern in the next stage. The reason for the order is that
value is the part that still reads when everything else is stripped away, not that color is
optional.

Set the two ends of the value range first: the darkest dark and the lightest light, placed
as actual shapes on the canvas. The value study image does this too, which is why it contains
pure black and pure white. Everything else is judged between those two anchors, so placing
both before any midtone is what keeps the middle of the range honest.
The brightest accents, the small sharp highlights, still come last; that's a different thing
from establishing the lightest mass, and confusing the two is what leads to a painting with
no bright note left to give.

Reading the value range: a camera does not see value the way an eye does. Shadows crush
toward black and highlights blow toward white in a way the actual scene never did. The
measured value range says what is in this photo, not what was in front of the camera. If the
darkest pixel is well above 0 or the lightest well below 255, the photo never reaches true
black or true white, and forcing them into the painting invents contrast the scene did not
have.

Beginner mistake: blocking in an object's entire form as one solid value instead of first
finding the shadow shapes inside it. This flattens the form and loses depth. It is not about
anatomy specifically, it is about the proper form of any object.

How to check the block-in. Squint at the reference photo. Squinting works by throwing away
detail so only the big value masses survive, which is exactly the information this stage
needs; photographing the painting and looking at it as a phone thumbnail does the same job
and is easier to compare side by side. Aim for about as many masses as the value study shows,
3 for the roughest pass and 5 as it refines, rather than an arbitrary number. Then check the
negative space between the shadow shapes: a shape that is the right size in the wrong place
reads as wrong just as fast as a shape that is the wrong size. If the photo has a clear
subject, recognizing it from the block-in alone is a good sign. If the photo is texture or
atmosphere rather than a nameable subject, that test does not apply, and accurate shapes and
spacing are the whole test.

Rendering one shape's color to completion before touching the shape beside it makes the two
hard to read as part of the same painting. Work rough color across the whole canvas at once
instead, the same way the value pass did, so the whole piece gets more resolved together
rather than one corner finishing while another has not been started.

On the suggested tubes: they are the closest match from a limited reference palette to the
photo's own dominant colors, meant as starting points for mixing at this stage, not as colors
to use straight from the tube. A large distance number means the photo color sits far from
any tube at full strength, which is normal, since most of a photograph is tints and mixtures
rather than pure pigment.

Midtones

Then the midtones, between the anchors. Without the anchors already down, a midtone can end
up too dark or too light because there's nothing yet to judge it against, and shadows in
particular get lost and muddied.

A painting does not have to match a photo's color one to one. It is usually better to
establish a mood through unified color. The judgment call is emphasis: put the most color and
the most detail where the eye should go, and let the areas that are not the focus fade into
simpler, more unified color. A distant mountain might be one flat color simply because it is
far away and is not the subject. Color contributes to emphasis the same way value does.

Edges start to matter here. A hard edge, a crisp division between two shapes, pulls the eye.
A soft edge, a gradual transition, lets a shape sit back and recede. Nothing needs a hard
edge yet, but this is the stage to start deciding which edges will eventually be hard and
which stay soft, because that decision is a large part of what controls where a viewer looks.

Temperature becomes a live factor here too. The usual starting assumption is that light is
warmer and shadow and distance are cooler, but that is a default to test, not a rule to
apply: overcast light, indoor light, backlighting, and haze can all flip it. Temperature is
also relative rather than absolute. A whole scene can measure cool, in which case the warmer
passages are only warmer than what sits beside them, and painting the foreground genuinely
warm to manufacture contrast would be wrong for that photo. When the measured temperature
disagrees with the first impression, trust the measurement: atmospheric haze in particular
makes a scene read warm to the eye while measuring distinctly cool.

What has to be locked in by the end of this stage: the value structure and the rough color
relationships. What is still allowed to be messy: edges and small detail.

Full detail

Before starting detail, step back and squint at the whole piece: are the proportions readable,
do the color relationships hold together, is the warmth and coolness doing what it should.
This stage is a final check on every earlier stage before committing to rendering.

Most common beginner mistake: rushing into detail produces hyper-focus on one part instead of
the whole image, and proportion errors surface here because the earlier stages were never
fully checked. It also creates attachment. Once real time has gone into detailing something,
there is resistance to fixing the underlying proportion problem even after seeing it, because
fixing it means giving up finished work.

Knowing when to stop. Overworking is the most common way a beginner painting goes wrong at
this stage, and it happens because there is no obvious signal to stop. Useful checks: the
painting reads correctly at thumbnail size, adding detail is no longer changing how it reads
from across the room, and the areas that were meant to stay simple are still simple. Detail
belongs where the emphasis is. If detail has spread evenly across the whole canvas, the
emphasis decided in the color stage has been undone.

Throughline

Each stage exists to lock down a reference the next stage is judged against: the
drawing before any paint, the value anchors before the midtones, the block-in before detail.
Skipping a stage does not just risk that stage, it removes the reference every later stage
needed. Step back and squint at every stage; it is the same check each time, applied to more
information.
"""

# The exact labels the model has to use are built from imaging.STAGE_CAPTIONS rather than
# retyped here, so the filmstrip and the writeup can never drift into two numbering schemes.
#
# The "name what you see" requirement is not padding. The rubric on its own is addressed to
# a painter in the abstract and never once says to look at the photo, so a model can satisfy
# it completely while writing something that would fit any image. That failure is the exact
# thing this session is supposed to verify against, so the instruction to be specific has to
# be explicit rather than implied.
#
# The no-preamble rule names canvas preparation specifically because the model mirrors the
# rubric's own structure, and every step it writes has to have an image beside it. A
# "Before starting" section is the one thing the rubric can teach that the filmstrip cannot
# show. It is phrased against what the rubric describes rather than against a step label, so
# it survives the rubric being reorganized without going stale the way its v2 wording did.
OUTPUT_INSTRUCTIONS = (
    "Look at the attached photo and write a step-by-step guide for painting that specific "
    "photo, following the rubric above and the measured stats below.\n\n"
    "Every step must refer to what is actually in this image: name the subject and the "
    "things in it, say where they sit in the frame, and describe the actual light, the "
    "actual shadow shapes, and the actual colors present. A reader should not be able to "
    "swap in a different photograph and have the guide still make sense. Do not give advice "
    "that would apply to any photo.\n\n"
    "Where the measured stats are relevant, use them and say what they mean for this photo "
    "rather than restating the number.\n\n"
    "Write exactly 4 steps, each with one of these exact labels, in this order: "
    + ", ".join(f'"{caption}"' for caption in STAGE_CAPTIONS) + ". Do not add a preamble, "
    "an introduction, a title, or any section before the first label. Anything the rubric "
    "describes as happening before the paint goes on, toning the ground included, belongs "
    "inside the first step rather than in front of it. Do not add a closing summary after "
    "the last step.\n\n"
    "Hard limit of 400 words for the whole thing, which is about 100 words per step. This "
    "is a ceiling, not a target to pad toward. Being specific about this photo matters more "
    "than covering every point in the rubric, so choose the parts of the rubric that this "
    "particular image actually calls for and leave the rest out. Address the painter "
    "directly and keep it practical."
)


def build_prompt(value_range, temperature):
    """Assemble the full prompt text: rubric, this photo's measured stats, and the
    output-shape instructions.

    Stats are sent as raw numbers with the axis explained, never as an invented category
    (a "compressed range" or "warm" label behind a threshold nobody measured). That's the
    same mistake the paint match nearly made in S7 with an unmeasured Delta E 12
    "approximate" cutoff; the fix there was showing the number instead of a bucket, and it
    applies here too.

    Args:
        value_range: (min, max) tuple from imaging.value_range, 0-255 scale.
        temperature: float from imaging.dominant_temperature.

    Returns:
        The full prompt text, sent as the text content block alongside the base64 photo.
    """
    darkest, lightest = value_range
    return (
        f"{RUBRIC}\n"
        "Measured stats for this specific photo, not a general description:\n"
        f"- Value range: darkest pixel {darkest}, lightest pixel {lightest}, on a 0-255 "
        "scale where 0 is true black and 255 is true white. This is the actual range "
        "present in the photo, not an assumption that it spans the full scale.\n"
        f"- Dominant temperature: {temperature:.1f}, a share-weighted average of the "
        "photo's own palette on the Lab b axis (blue to yellow; positive sits toward "
        "yellow, negative sits toward blue). This is a measurement of the photo's actual "
        "pixels, not a subjective read, and it can disagree with how the photo looks at a "
        "glance.\n\n"
        f"{OUTPUT_INSTRUCTIONS}"
    )
