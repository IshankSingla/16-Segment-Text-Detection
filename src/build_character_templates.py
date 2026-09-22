"""
Phase 1 - Automatic Character Template Builder

Builds character templates from the current dataset.

Method 1 (16-segment decoder) is used only to identify
which character is present.

Method 2 recognition itself uses image/template similarity.
"""

from pathlib import Path

import cv2
import numpy as np

from segment_decoder import (
    calibrate_multiple_cell_starts,
    decode_frame,
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
# TEMPLATE SIZE
# ============================================================

TEMPLATE_WIDTH = 50
TEMPLATE_HEIGHT = 76


# ============================================================
# DISPLAY MASK
# ============================================================

def create_display_mask(image):
    """
    Create a binary image containing illuminated display pixels.
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


# ============================================================
# CHARACTER NORMALIZATION
# ============================================================

def normalize_character(cell):
    """
    Normalize a character image.

    Steps:
        1. Find illuminated pixels.
        2. Crop to bounding box.
        3. Add padding.
        4. Resize to common template size.
    """

    if cell is None or cell.size == 0:
        return None

    points = cv2.findNonZero(cell)

    if points is None:
        return None

    x, y, w, h = cv2.boundingRect(points)

    if w < 5 or h < 5:
        return None

    cropped = cell[
        y:y + h,
        x:x + w,
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
# CELL EXTRACTION
# ============================================================

def extract_cell(
    mask,
    cell_x,
):
    """
    Extract one character cell.
    """

    x1 = int(cell_x)

    x2 = min(
        x1 + 55,
        mask.shape[1],
    )

    if x1 >= mask.shape[1]:
        return None

    return mask[
        0:mask.shape[0],
        x1:x2,
    ]


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("AUTOMATIC CHARACTER TEMPLATE BUILDER")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Load current dataset
    # --------------------------------------------------------

    images = sorted(
        TEXT_DIR.glob("*.bmp")
    )

    if not images:

        raise RuntimeError(
            f"No BMP files found in {TEXT_DIR}"
        )

    print(
        f"Found {len(images)} BMP frames."
    )

    print()

    # --------------------------------------------------------
    # Automatically detect one or more display geometries.
    #
    # A combined folder may contain datasets with different
    # horizontal display alignment. Each frame is assigned to
    # its automatically detected calibration group.
    # --------------------------------------------------------

    calibration = calibrate_multiple_cell_starts(
        images
    )

    frame_groups = calibration["frame_groups"]
    calibrations = calibration["calibrations"]

    print(
        f"Using {len(calibrations)} automatic "
        f"calibration group(s)."
    )

    print()

    # --------------------------------------------------------
    # Remove old templates
    # --------------------------------------------------------

    TEMPLATE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    old_templates = list(
        TEMPLATE_DIR.glob("*.png")
    )

    for path in old_templates:
        path.unlink()

    print(
        f"Removed {len(old_templates)} old templates."
    )

    print()

    # --------------------------------------------------------
    # Candidate storage
    #
    # character -> list of:
    #
    # (
    #     normalized image,
    #     frame number,
    #     cell number,
    #     quality,
    #     confidence
    # )
    # --------------------------------------------------------

    candidates = {}

    # --------------------------------------------------------
    # Process every frame
    # --------------------------------------------------------

    for frame_number, image_path in enumerate(
        images,
        start=1,
    ):

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                f"WARNING: Could not read "
                f"{image_path.name}"
            )

            continue

        # ----------------------------------------------------
        # Method 1 identifies the characters.
        #
        # IMPORTANT:
        # Use the same threshold that made the standalone
        # segment decoder work.
        # ----------------------------------------------------

        group_id = frame_groups[frame_number - 1]

        cell_starts = calibrations.get(
            group_id,
            [],
        )

        results = decode_frame(
            image,
            threshold=0.30,
            cell_starts=cell_starts,
        )

        # ----------------------------------------------------
        # Create binary display mask.
        # ----------------------------------------------------

        mask = create_display_mask(
            image
        )

        # ----------------------------------------------------
        # Examine each cell.
        # ----------------------------------------------------

        for cell_number, result in enumerate(
            results
        ):

            character = result.character

            # Ignore blank and uncertain results.
            if (
                not character
                or character == " "
                or character == "?"
            ):
                continue

            # Safety check.
            if cell_number >= len(
                cell_starts
            ):
                continue

            cell_x = cell_starts[
                cell_number
            ]

            cell = extract_cell(
                mask,
                cell_x,
            )

            if cell is None:
                continue

            # ------------------------------------------------
            # Pixel quality.
            # ------------------------------------------------

            quality = cv2.countNonZero(
                cell
            )

            # Ignore weak/clipped characters.
            if quality < 500:
                continue

            normalized = normalize_character(
                cell
            )

            if normalized is None:
                continue

            # ------------------------------------------------
            # Automatically create character entry.
            # ------------------------------------------------

            if character not in candidates:
                candidates[character] = []

            candidates[character].append(
                (
                    normalized,
                    frame_number,
                    cell_number,
                    quality,
                    result.confidence,
                )
            )

    # ========================================================
    # DISCOVERED CHARACTERS
    # ========================================================

    print()
    print("=" * 70)
    print("CHARACTERS DISCOVERED")
    print("=" * 70)
    print()

    discovered = sorted(
        candidates.keys()
    )

    print(
        " ".join(discovered)
    )

    print()

    # ========================================================
    # SELECT BEST TEMPLATE
    # ========================================================

    print("=" * 70)
    print("SELECTING BEST TEMPLATES")
    print("=" * 70)
    print()

    saved_count = 0

    for character in discovered:

        character_candidates = candidates[
            character
        ]

        if not character_candidates:
            continue

        # ----------------------------------------------------
        # Select the strongest example.
        # ----------------------------------------------------

        best = max(
            character_candidates,
            key=lambda item: (
                item[3],
                item[4],
            ),
        )

        (
            template,
            frame_number,
            cell_number,
            quality,
            confidence,
        ) = best

        output_path = (
            TEMPLATE_DIR
            / f"{character}.png"
        )

        cv2.imwrite(
            str(output_path),
            template,
        )

        print(
            f"Saved {character}.png "
            f"| frame={frame_number} "
            f"| cell={cell_number + 1} "
            f"| quality={quality} "
            f"| confidence={confidence:.2f}"
        )

        saved_count += 1

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("TEMPLATE BUILD COMPLETE")
    print("=" * 70)
    print()

    print(
        f"Frames processed : {len(images)}"
    )

    print(
        f"Characters found : {len(discovered)}"
    )

    print(
        f"Templates created: {saved_count}"
    )

    print()

    print(
        "Templates saved in:"
    )

    print(
        TEMPLATE_DIR
    )

    print()


if __name__ == "__main__":
    main()