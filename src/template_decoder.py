"""
Method 2: Image Template Matching Decoder

This decoder recognizes characters by comparing their actual
pixel appearance with stored character templates.

It does NOT use:
    - 16-segment patterns
    - CHARACTER_SEGMENTS
    - Hamming distance

It uses image similarity.

Phase 1 improvements:
    - Supports any number of BMP frames.
    - Automatically calibrates character-cell positions.
    - Uses the same geometry as Method 1.
    - Loads templates generated from the current dataset.
"""


from pathlib import Path

import cv2
import numpy as np

from segment_decoder import (
    calibrate_multiple_cell_starts,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEXT_DIR = PROJECT_ROOT / "data" / "text"

TEMPLATE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "templates"
)


# ============================================================
# DISPLAY GEOMETRY
# ============================================================

CELL_WIDTH = 55
CELL_HEIGHT = 86


# ============================================================
# TEMPLATE SIZE
# ============================================================

TEMPLATE_WIDTH = 50
TEMPLATE_HEIGHT = 76


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def create_display_mask(image):
    """
    Convert the display image into a binary LED mask.
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    _, saturation, value = cv2.split(
        hsv
    )

    mask = (
        (saturation > 80)
        & (value > 80)
    ).astype(
        np.uint8
    ) * 255

    return mask


# ============================================================
# CHARACTER NORMALIZATION
# ============================================================

def normalize_character(cell):
    """
    Normalize a character image.

    Steps:
        1. Find illuminated pixels.
        2. Crop to their bounding box.
        3. Add padding.
        4. Resize to the template size.
    """

    if cell is None or cell.size == 0:
        return None

    points = cv2.findNonZero(
        cell
    )

    if points is None:
        return None

    x, y, w, h = cv2.boundingRect(
        points
    )

    if w < 5 or h < 5:
        return None

    cropped = cell[
        y:y + h,
        x:x + w
    ]

    padding = 5

    cropped = cv2.copyMakeBorder(
        cropped,
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_CONSTANT,
        value=0,
    )

    normalized = cv2.resize(
        cropped,
        (
            TEMPLATE_WIDTH,
            TEMPLATE_HEIGHT,
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    return normalized


# ============================================================
# LOAD TEMPLATES
# ============================================================

def load_templates():
    """
    Load all PNG character templates.
    """

    templates = {}

    for path in sorted(
        TEMPLATE_DIR.glob("*.png")
    ):

        character = path.stem.upper()

        image = cv2.imread(
            str(path),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            continue

        templates[
            character
        ] = image

    if not templates:

        raise RuntimeError(
            f"No templates found in "
            f"{TEMPLATE_DIR}"
        )

    return templates


# ============================================================
# EXTRACT CELL
# ============================================================

def extract_cell(
    mask,
    cell_x,
):
    """
    Extract one character cell using the
    automatically calibrated cell position.
    """

    x1 = int(round(cell_x))

    x2 = min(
        x1 + CELL_WIDTH,
        mask.shape[1],
    )

    if x1 >= mask.shape[1]:
        return None

    return mask[
        0:min(CELL_HEIGHT, mask.shape[0]),
        x1:x2,
    ]


# ============================================================
# IMAGE SIMILARITY
# ============================================================

def calculate_similarity(
    image_a,
    image_b,
):
    """
    Calculate normalized image similarity.

    Returns:
        0.0 -> completely different
        1.0 -> identical
    """

    if (
        image_a is None
        or image_b is None
    ):
        return 0.0

    if image_a.shape != image_b.shape:

        image_b = cv2.resize(
            image_b,
            (
                image_a.shape[1],
                image_a.shape[0],
            ),
            interpolation=cv2.INTER_NEAREST,
        )

    # Convert to floating point.
    a = image_a.astype(
        np.float32
    ) / 255.0

    b = image_b.astype(
        np.float32
    ) / 255.0

    # Mean absolute difference.
    difference = np.mean(
        np.abs(a - b)
    )

    similarity = 1.0 - difference

    return float(
        max(
            0.0,
            min(
                1.0,
                similarity,
            ),
        )
    )


# ============================================================
# DECODE CHARACTER
# ============================================================

def decode_character(
    cell,
    templates,
):
    """
    Find the template with the highest image similarity.

    Returns:
        character
        similarity
    """

    normalized = normalize_character(
        cell
    )

    if normalized is None:

        return (
            " ",
            1.0,
        )

    best_character = "?"
    best_similarity = -1.0

    for character, template in (
        templates.items()
    ):

        similarity = calculate_similarity(
            normalized,
            template,
        )

        if similarity > best_similarity:

            best_similarity = (
                similarity
            )

            best_character = character

    return (
        best_character,
        best_similarity,
    )


# ============================================================
# DECODE FRAME
# ============================================================

def decode_frame(
    image,
    templates,
    cell_starts,
    min_similarity=0.65,
):
    """
    Decode all calibrated character positions
    in one frame.
    """

    mask = create_display_mask(
        image
    )

    decoded = []

    confidences = []

    for cell_x in cell_starts:

        cell = extract_cell(
            mask,
            cell_x,
        )

        if cell is None:

            decoded.append("?")
            confidences.append(0.0)

            continue

        # ----------------------------------------------------
        # Determine whether this cell is blank.
        # ----------------------------------------------------

        illuminated = cv2.countNonZero(
            cell
        )

        if illuminated < 100:

            decoded.append(" ")
            confidences.append(1.0)

            continue

        character, similarity = (
            decode_character(
                cell,
                templates,
            )
        )

        # ----------------------------------------------------
        # Reject poor matches.
        # ----------------------------------------------------

        if similarity < min_similarity:

            decoded.append("?")

        else:

            decoded.append(
                character
            )

        confidences.append(
            similarity
        )

    return (
        "".join(decoded).strip(),
        confidences,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("METHOD 2 — IMAGE TEMPLATE MATCHING")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Load templates.
    # --------------------------------------------------------

    templates = load_templates()

    print(
        f"Loaded {len(templates)} character templates."
    )

    print(
        "Characters:",
        " ".join(
            sorted(
                templates.keys()
            )
        ),
    )

    print()

    # --------------------------------------------------------
    # Load current dataset.
    # --------------------------------------------------------

    images = sorted(
        TEXT_DIR.glob("*.bmp")
    )

    if not images:

        raise RuntimeError(
            f"No BMP images found in "
            f"{TEXT_DIR}"
        )

    print(
        f"Found {len(images)} frames."
    )

    print()

    # --------------------------------------------------------
    # IMPORTANT:
    # Detect one or more display geometries.
    #
    # The template builder uses the same multi-calibration
    # system, so every frame is decoded using the geometry
    # appropriate for its dataset/alignment.
    # --------------------------------------------------------

    calibration = calibrate_multiple_cell_starts(
        images
    )

    frame_groups = calibration["frame_groups"]
    calibrations = calibration["calibrations"]

    print(
        f"Using {len(calibrations)} "
        f"automatic calibration group(s)."
    )

    print()

    # --------------------------------------------------------
    # Decode every frame.
    # --------------------------------------------------------

    for frame_number, image_path in enumerate(
        images,
        start=1,
    ):

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        group_id = frame_groups[frame_number - 1]

        cell_starts = calibrations.get(
            group_id,
            [],
        )

        text, confidences = (
            decode_frame(
                image,
                templates,
                cell_starts,
            )
        )

        print(
            f"Frame {frame_number:03d}: "
            f"{text}"
        )

    print()
    print("=" * 70)
    print("METHOD 2 COMPLETE")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()