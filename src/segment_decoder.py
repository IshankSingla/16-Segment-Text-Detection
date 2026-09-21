"""
Method 1: 16-Segment Display Decoder

This module detects characters by checking which of the
16 physical display segments are illuminated.

No OCR or machine-learning model is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


# ------------------------------------------------------------
# DISPLAY GEOMETRY
# ------------------------------------------------------------

# Approximate character-cell starting positions measured from
# the supplied 825 x 86 display frames.
#
# There are 13 character positions across the display.

CELL_STARTS = [
    18,
    80,
    141,
    202,
    264,
    326,
    387,
    448,
    510,
    572,
    633,
    694,
    756,
]

# Character width.
CELL_WIDTH = 55

# The segment geometry inside a character cell.
#
# Coordinates are relative to the character-cell origin.
#
# The 16 segments are:
#
# a1, a2  -> top-left / top-right
# b, c    -> upper-right / lower-right
# d1, d2  -> bottom-right / bottom-left
# e, f    -> lower-left / upper-left
# g1, g2  -> middle-left / middle-right
# h, i    -> upper diagonals
# j, k    -> lower diagonals
# l, m    -> upper/lower center vertical
#
# These coordinates were chosen from the actual supplied
# display geometry.

SEGMENT_NAMES = [
    "a1",
    "a2",
    "b",
    "c",
    "d1",
    "d2",
    "e",
    "f",
    "g1",
    "g2",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
]


SEGMENTS = [
    # Top
    ((8, 10), (27, 10)),       # a1
    ((29, 10), (50, 10)),       # a2

    # Right vertical
    ((52, 14), (52, 36)),       # b
    ((52, 46), (52, 70)),       # c

    # Bottom
    ((29, 72), (50, 72)),       # d1
    ((8, 72), (27, 72)),        # d2

    # Left vertical
    ((3, 46), (3, 70)),         # e
    ((3, 14), (3, 36)),         # f

    # Middle
    ((8, 42), (27, 42)),         # g1
    ((29, 42), (50, 42)),        # g2

    # Upper diagonals
    ((8, 14), (27, 36)),         # h
    ((50, 14), (32, 36)),        # i

    # Lower diagonals
    ((8, 70), (27, 46)),         # j
    ((50, 70), (32, 46)),        # k

    # Center vertical
    ((28, 14), (28, 36)),        # l
    ((28, 46), (28, 70)),        # m
]


# ------------------------------------------------------------
# CHARACTER PATTERNS
# ------------------------------------------------------------

# Each character is represented by the segments that should
# be illuminated.
#
# These patterns are based on the actual 16-segment display
# used in the supplied images.
#
# Characters needed by the supplied messages are explicitly
# defined first. Additional common characters are included
# for reuse.

CHARACTER_SEGMENTS = {

    # --------------------------------------------------------
    # A-Z 16-SEGMENT CHARACTER DEFINITIONS
    # --------------------------------------------------------

    "A": {
        "a1", "a2",
        "b", "c",
        "e", "f",
        "g1", "g2",
    },

    "B": {
        "a1", "a2",
        "b", "c",
        "d1", "d2",
        "g2",
        "l", "m",
    },

    "C": {
        "a1", "a2",
        "d1", "d2",
        "e", "f",
    },

    "D": {
        # The supplied 16-segment display renders D with
        # the right-side middle segment and lower-left diagonal
        # in addition to the main D structure.
        "a1", "a2",
        "b", "c",
        "d1", "d2",
        "g2",
        "j",
        "l", "m",
    },

    "E": {
        "a1", "a2",
        "d1", "d2",
        "e", "f",
        "g1", "g2",
    },

    "F": {
        "a1", "a2",
        "e", "f",
        "g1", "g2",
    },

    "G": {
        "a1", "a2",
        "c",
        "d1", "d2",
        "e", "f",
        "g1", "g2",
    },

    "H": {
        "b", "c",
        "e", "f",
        "g1", "g2",
    },

    "I": {
        "a1", "a2",
        "d1", "d2",
        "l", "m",
    },

    "J": {
        "b", "c",
        "d1", "d2",
        "e",
    },

    "K": {
        "e", "f",
        "g1",
        "i", "k",
    },

    "L": {
        "d1", "d2",
        "e", "f",
    },

    "M": {
        "b", "c",
        "e", "f",
        "h", "i",
    },

    "N": {
        "e", "f",
        "b", "c",
        "h", "k",
    },

    "O": {
        "a1", "a2",
        "b", "c",
        "d1", "d2",
        "e", "f",
    },

    "P": {
        "a1", "a2",
        "b",
        "e", "f",
        "g1", "g2",
    },

    "Q": {
        "a1", "a2",
        "b", "c",
        "d1", "d2",
        "e", "f",
        "k",
    },

    "R": {
        "a1", "a2",
        "b",
        "e", "f",
        "g1", "g2",
        "k",
    },

    "S": {
        "a1", "a2",
        "c",
        "d1", "d2",
        "g1", "g2",
        "f",
    },

    "T": {
        "a1", "a2",
        "l", "m",
    },

    "U": {
        "b", "c",
        "d1", "d2",
        "e", "f",
    },

    "V": {
        "e", "f",
        "c",
        "j", "k",
    },

    "W": {
        "b", "c",
        "e", "f",
        "j", "k",
    },

    "X": {
        "h", "i",
        "j", "k",
    },

    "Y": {
        "h", "i",
        "m",
    },

    "Z": {
        "a1", "a2",
        "i", "j",
        "d1", "d2",
    },
}

# ------------------------------------------------------------
# DATA STRUCTURE
# ------------------------------------------------------------

@dataclass
class CharacterResult:
    """
    Result for one character cell.
    """

    character: str
    pattern: tuple[int, ...]
    confidence: float


# ------------------------------------------------------------
# IMAGE PREPROCESSING
# ------------------------------------------------------------

def create_display_mask(
    image: np.ndarray,
) -> np.ndarray:
    """
    Convert the display image into a binary LED mask.

    The display LEDs are bright and saturated blue/cyan,
    while the background is dark.
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    _, saturation, value = cv2.split(hsv)

    mask = (
        (saturation > 80)
        & (value > 80)
    ).astype(
        np.uint8
    ) * 255

    return mask


# ------------------------------------------------------------
# AUTOMATIC DATASET CALIBRATION
# ------------------------------------------------------------

def _find_projection_peaks(
    projection: np.ndarray,
    threshold: float,
) -> list[float]:
    """Find centers of strong vertical illuminated regions."""

    active = np.where(
        projection >= threshold
    )[0]

    if len(active) == 0:
        return []

    runs = []

    start = int(active[0])
    previous = start

    for x in active[1:]:
        x = int(x)

        if x > previous + 1:
            runs.append((start, previous))
            start = x

        previous = x

    runs.append((start, previous))

    centers = []

    for start, end in runs:
        width = end - start + 1

        if width >= 2:
            centers.append(
                (start + end) / 2.0
            )

    return centers


def _grid_fit_score(
    centers: list[float],
    pitch: float,
    phase: float,
    tolerance: float = 4.0,
) -> tuple[int, float]:
    """Score how well detected vertical peaks fit the 16-segment grid."""

    if not centers:
        return 0, float("-inf")

    expected = []

    # A character cell has vertical segments around x+3 and x+52.
    # These are the two strongest repeated vertical features.
    for index in range(20):
        cell_x = phase + index * pitch

        expected.append(cell_x + 3)
        expected.append(cell_x + 52)

    expected = np.asarray(expected)

    distances = np.min(
        np.abs(
            np.asarray(centers)[:, None]
            - expected[None, :]
        ),
        axis=1,
    )

    matches = distances <= tolerance

    count = int(np.count_nonzero(matches))

    # Reward close matches while still preferring more matches.
    closeness = float(
        np.sum(
            np.exp(
                -(
                    distances / 2.0
                ) ** 2
            )
        )
    )

    return count, closeness


def calibrate_cell_starts(
    image_paths,
) -> list[float]:
    """
    Automatically estimate horizontal character-cell positions
    for the current dataset.

    The physical display is assumed to be the same 16-segment
    display, but the text may start at a different horizontal
    position.

    Calibration uses strong vertical LED segments to estimate:
        - character pitch
        - horizontal grid phase

    The original CELL_STARTS remain the fallback if calibration
    cannot obtain enough evidence.
    """

    if not image_paths:
        return CELL_STARTS.copy()

    # Use several frames so that blank/partial scrolling frames
    # do not determine the geometry by themselves.
    sample_paths = list(image_paths)[:30]

    all_centers = []

    for image_path in sample_paths:

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        mask = create_display_mask(
            image
        )

        projection = np.count_nonzero(
            mask > 0,
            axis=0,
        )

        # Vertical segments occupy much more vertical area than
        # horizontal segments, so a relatively high threshold
        # isolates them.
        threshold = max(
            20.0,
            mask.shape[0] * 0.45,
        )

        centers = _find_projection_peaks(
            projection,
            threshold,
        )

        all_centers.extend(
            centers
        )

    if len(all_centers) < 4:
        print(
            "Automatic calibration could not find "
            "enough vertical segments."
        )
        print(
            "Using original CELL_STARTS."
        )
        return CELL_STARTS.copy()

    # --------------------------------------------------------
    # Estimate the physical character pitch.
    #
    # Search a narrow range around the original display pitch.
    # This makes calibration robust without changing the actual
    # 16-segment geometry.
    # --------------------------------------------------------

    best_pitch = None
    best_phase = None
    best_count = -1
    best_closeness = float("-inf")

    for pitch in np.linspace(
        58.0,
        65.0,
        141,
    ):

        for phase in np.linspace(
            0.0,
            pitch,
            121,
        ):

            count, closeness = _grid_fit_score(
                all_centers,
                pitch,
                phase,
            )

            if (
                count > best_count
                or (
                    count == best_count
                    and closeness > best_closeness
                )
            ):
                best_count = count
                best_closeness = closeness
                best_pitch = pitch
                best_phase = phase

    if best_pitch is None:
        return CELL_STARTS.copy()

    # --------------------------------------------------------
    # Convert grid phase into cell starts.
    #
    # The grid phase is the beginning of cell 0.
    # Generate enough cells to cover the image.
    # --------------------------------------------------------

    # Estimate how many cells fit in the image.
    image_width = 0

    first_image = cv2.imread(
        str(sample_paths[0])
    )

    if first_image is not None:
        image_width = first_image.shape[1]

    if image_width <= 0:
        return CELL_STARTS.copy()

    cell_starts = []

    # Include cells that overlap the image.
    index_min = int(
        np.floor(
            (-CELL_WIDTH - best_phase)
            / best_pitch
        )
    )

    index_max = int(
        np.ceil(
            (image_width - best_phase)
            / best_pitch
        )
    )

    for index in range(
        index_min,
        index_max + 1,
    ):

        cell_x = (
            best_phase
            + index * best_pitch
        )

        # Keep complete character cells only.
        if (
            cell_x >= 0
            and cell_x + CELL_WIDTH <= image_width
        ):
            cell_starts.append(
                round(float(cell_x), 2)
            )

    # Keep the expected 13 display positions when the detected
    # pitch is consistent with the supplied display.
    expected_count = round(
        image_width / best_pitch
    )

    if expected_count < 10 or expected_count > 16:
        print(
            "Calibration produced an unusual number "
            "of cells. Using original CELL_STARTS."
        )
        return CELL_STARTS.copy()

    print()
    print("=" * 70)
    print("AUTOMATIC DISPLAY CALIBRATION")
    print("=" * 70)
    print(
        f"Frames inspected : {len(sample_paths)}"
    )
    print(
        f"Vertical samples : {len(all_centers)}"
    )
    print(
        f"Estimated pitch  : {best_pitch:.2f} px"
    )
    print(
        f"Estimated phase  : {best_phase:.2f} px"
    )
    print(
        f"Detected cells   : {len(cell_starts)}"
    )
    print(
        f"Cell starts      : {cell_starts}"
    )
    print("=" * 70)
    print()

    return cell_starts


# ------------------------------------------------------------
# SEGMENT SAMPLING
# ------------------------------------------------------------

def sample_segment(
    mask: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    cell_x: int,
    y_offset: int = 5,
    radius: int = 2,
) -> float:
    """
    Calculate the percentage of an individual segment that
    contains illuminated pixels.

    Returns:
        value between 0 and 1.
    """

    x1 = p1[0] + cell_x
    y1 = p1[1] + y_offset

    x2 = p2[0] + cell_x
    y2 = p2[1] + y_offset

    length = int(
        math.hypot(
            x2 - x1,
            y2 - y1,
        )
    ) + 1

    xs = np.linspace(
        x1,
        x2,
        length,
    )

    ys = np.linspace(
        y1,
        y2,
        length,
    )

    samples = []

    for x, y in zip(xs, ys):

        xi = int(round(x))
        yi = int(round(y))

        x_min = max(
            0,
            xi - radius,
        )

        x_max = min(
            mask.shape[1],
            xi + radius + 1,
        )

        y_min = max(
            0,
            yi - radius,
        )

        y_max = min(
            mask.shape[0],
            yi + radius + 1,
        )

        region = mask[
            y_min:y_max,
            x_min:x_max,
        ]

        if region.size:
            samples.append(
                np.mean(region > 0)
            )

    if not samples:
        return 0.0

    return float(
        np.mean(samples)
    )


def get_segment_scores(
    mask: np.ndarray,
    cell_x: int,
) -> list[float]:
    """
    Measure all 16 segments in one character cell.
    """

    scores = []

    for p1, p2 in SEGMENTS:

        score = sample_segment(
            mask,
            p1,
            p2,
            cell_x,
        )

        scores.append(
            score
        )

    return scores


# ------------------------------------------------------------
# PATTERN CREATION
# ------------------------------------------------------------

def scores_to_pattern(
    scores: list[float],
    threshold: float = 0.30,
) -> tuple[int, ...]:
    """
    Convert segment brightness scores into an ON/OFF pattern.

    A single global threshold is not ideal for this display because
    the middle horizontal segments can be slightly dimmer, while
    diagonal segments can receive light spill from neighbouring
    segments.

    Therefore:
        - normal segments use the supplied threshold;
        - g1/g2 use a slightly lower threshold;
        - diagonal segments use a slightly higher threshold.
    """

    if len(scores) != len(SEGMENTS):
        raise ValueError(
            "Expected 16 segment scores."
        )

    thresholds = [
        threshold,  # a1
        threshold,  # a2
        threshold,  # b
        threshold,  # c
        threshold,  # d1
        threshold,  # d2
        threshold,  # e
        threshold,  # f
        0.27,       # g1
        0.27,       # g2
        0.35,       # h
        0.35,       # i
        0.35,       # j
        0.35,       # k
        threshold,  # l
        threshold,  # m
    ]

    return tuple(
        1 if score >= segment_threshold else 0
        for score, segment_threshold
        in zip(scores, thresholds)
    )


def character_to_pattern(
    character: str,
) -> tuple[int, ...]:
    """
    Convert a character into its expected 16-segment pattern.
    """

    character = character.upper()

    active_segments = CHARACTER_SEGMENTS.get(
        character,
        set(),
    )

    return tuple(
        1 if name in active_segments else 0
        for name in SEGMENT_NAMES
    )


# ------------------------------------------------------------
# CHARACTER MATCHING
# ------------------------------------------------------------

def pattern_distance(
    observed: tuple[int, ...],
    expected: tuple[int, ...],
) -> float:
    """
    Calculate normalized Hamming distance between two patterns.
    """

    if len(observed) != len(expected):
        raise ValueError(
            "Patterns must have the same length."
        )

    differences = sum(
        a != b
        for a, b in zip(
            observed,
            expected,
        )
    )

    return differences / len(
        observed
    )


def decode_character(
    pattern: tuple[int, ...],
) -> tuple[str, float]:
    """
    Find the character whose segment pattern is closest
    to the observed pattern.

    Returns:
        character
        confidence
    """

    best_character = "?"
    best_distance = float("inf")

    for character in CHARACTER_SEGMENTS:

        expected = character_to_pattern(
            character
        )

        distance = pattern_distance(
            pattern,
            expected,
        )

        if distance < best_distance:

            best_distance = distance
            best_character = character

    # Confidence is the inverse of normalized error.
    confidence = 1.0 - best_distance

    return (
        best_character,
        confidence,
    )


# ------------------------------------------------------------
# COMPLETE FRAME DECODING
# ------------------------------------------------------------

def decode_frame(
    image: np.ndarray,
    threshold: float = 0.30,
    cell_starts: list[float] | None = None,
) -> list[CharacterResult]:
    """
    Decode all character cells in one display frame.
    """

    mask = create_display_mask(
        image
    )

    results = []

    if cell_starts is None:
        cell_starts = CELL_STARTS

    for cell_x in cell_starts:

        scores = get_segment_scores(
            mask,
            cell_x,
        )

        pattern = scores_to_pattern(
            scores,
            threshold,
        )

        # Blank character detection.
        max_score = max(scores)

        if max_score < 0.30:

            result = CharacterResult(
                character=" ",
                pattern=pattern,
                confidence=1.0,
            )

        else:

            character, confidence = (
                decode_character(
                    pattern
                )
            )

            result = CharacterResult(
                character=character,
                pattern=pattern,
                confidence=confidence,
            )

        results.append(
            result
        )

    return results


def decode_text(
    image: np.ndarray,
    threshold: float = 0.30,
    min_confidence: float = 0.75,
    cell_starts: list[float] | None = None,
) -> str:
    """
    Decode the visible text.

    Characters with low confidence are represented as '?'.
    This prevents uncertain edge characters from being silently
    treated as reliable detections.
    """

    results = decode_frame(
        image,
        threshold,
        cell_starts,
    )

    decoded = []

    for result in results:

        # Preserve blank cells.
        if result.character == " ":

            decoded.append(" ")
            continue

        # Reject uncertain classifications.
        if result.confidence < min_confidence:

            decoded.append("?")

        else:

            decoded.append(
                result.character
            )

    return "".join(
        decoded
    ).strip()
if __name__ == "__main__":

    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    text_dir = project_root / "data" / "text"

    images = sorted(text_dir.glob("*.bmp"))

    if not images:
        raise RuntimeError("No BMP images found.")

    print()
    print("=" * 70)
    print("METHOD 1 - 16-SEGMENT DISPLAY DECODER")
    print("=" * 70)
    print()
    print(f"Total images: {len(images)}")
    print()

    # --------------------------------------------------------
    # Automatically calibrate the horizontal character grid
    # for the current dataset.
    # --------------------------------------------------------

    cell_starts = calibrate_cell_starts(
        images
    )

    for index, image_path in enumerate(images, start=1):

        image = cv2.imread(str(image_path))

        if image is None:
            print(
                f"Frame {index:03d}: "
                "ERROR - could not read image"
            )
            continue

        text = decode_text(
            image,
            threshold=0.30,
            cell_starts=cell_starts,
        )

        print(
            f"Frame {index:03d}: "
            f"{text}"
        )