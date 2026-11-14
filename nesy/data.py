"""Synthetic noisy digit images.

Each digit is a 5x7 bitmap glyph dropped at a random offset into a 9x7 canvas,
then corrupted with pixel flips and Gaussian noise. No download required, and
hard enough that perception has to be learned rather than looked up.
"""

from __future__ import annotations

import numpy as np

GLYPHS = {
    0: [" ### ", "#   #", "#  ##", "# # #", "##  #", "#   #", " ### "],
    1: ["  #  ", " ##  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### "],
    2: [" ### ", "#   #", "    #", "   # ", "  #  ", " #   ", "#####"],
    3: [" ### ", "#   #", "    #", "  ## ", "    #", "#   #", " ### "],
    4: ["   # ", "  ## ", " # # ", "#  # ", "#####", "   # ", "   # "],
    5: ["#####", "#    ", "#### ", "    #", "    #", "#   #", " ### "],
    6: [" ### ", "#    ", "#    ", "#### ", "#   #", "#   #", " ### "],
    7: ["#####", "    #", "   # ", "  #  ", "  #  ", "  #  ", "  #  "],
    8: [" ### ", "#   #", "#   #", " ### ", "#   #", "#   #", " ### "],
    9: [" ### ", "#   #", "#   #", " ####", "    #", "    #", " ### "],
}

GLYPH_H, GLYPH_W = 7, 5
HEIGHT, WIDTH = 9, 7
IMAGE_DIM = HEIGHT * WIDTH

_BITMAPS = np.array(
    [[[c == "#" for c in row] for row in GLYPHS[d]] for d in range(10)], dtype=float
)


def render_digit(digit: int, rng: np.random.Generator, noise: float = 0.2,
                 flip: float = 0.02) -> np.ndarray:
    """Return one noisy image of `digit` as a flat vector of length IMAGE_DIM."""
    canvas = np.zeros((HEIGHT, WIDTH))
    dy = rng.integers(0, HEIGHT - GLYPH_H + 1)
    dx = rng.integers(0, WIDTH - GLYPH_W + 1)
    canvas[dy:dy + GLYPH_H, dx:dx + GLYPH_W] = _BITMAPS[digit]
    flips = rng.random(canvas.shape) < flip
    canvas = np.where(flips, 1.0 - canvas, canvas)
    canvas = canvas + rng.normal(0.0, noise, canvas.shape)
    return canvas.ravel()


def make_pairs(n: int, rng: np.random.Generator, **kwargs):
    """Sample `n` pairs of digit images.

    Returns (X1, X2, d1, d2): two image arrays of shape (n, IMAGE_DIM) and the
    hidden digit labels. Training only ever sees a function of (d1, d2).
    """
    d1 = rng.integers(0, 10, n)
    d2 = rng.integers(0, 10, n)
    X1 = np.stack([render_digit(d, rng, **kwargs) for d in d1])
    X2 = np.stack([render_digit(d, rng, **kwargs) for d in d2])
    return X1, X2, d1, d2


def to_ascii(image: np.ndarray, threshold: float = 0.5) -> list[str]:
    """Render a flat image as lines of text for terminal display."""
    grid = image.reshape(HEIGHT, WIDTH)
    return ["".join("#" if v > threshold else "." for v in row) for row in grid]
