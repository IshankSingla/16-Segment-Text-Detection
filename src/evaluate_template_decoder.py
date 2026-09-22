import argparse
import re
import io
from pathlib import Path
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr

import cv2

from template_decoder import load_templates, decode_frame
from segment_decoder import calibrate_multiple_cell_starts


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "text"


# ============================================================
# COMMAND LINE ARGUMENT
# ============================================================

parser = argparse.ArgumentParser(
    description="Evaluate Method 2 template decoder for any target message."
)

parser.add_argument(
    "--target",
    required=True,
    help="Target message to evaluate."
)

args = parser.parse_args()

TARGET = args.target.upper().strip()


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Convert text to uppercase and remove spaces/special characters.
    """

    if text is None:
        return ""

    text = str(text).upper()

    return re.sub(r"[^A-Z0-9]", "", text)


NORMALIZED_TARGET = normalize_text(TARGET)


# ============================================================
# TIMESTAMP EXTRACTION
# ============================================================

def timestamp_from_filename(path):
    """
    Extract timestamp from filenames containing:

        YYYYMMDD_HHMMSSmmm

    Example:

        20250620_095916946_.bmp
        20250620_095916946_ (2).bmp

    Returns None when the filename has no supported timestamp.
    """

    stem = Path(path).stem

    # Remove Windows duplicate suffix.
    stem = re.sub(r"\s+\(\d+\)$", "", stem)

    match = re.search(
        r"(\d{8})_(\d{6})(\d{3})",
        stem
    )

    if not match:
        return None

    date_part = match.group(1)
    time_part = match.group(2)
    millis = match.group(3)

    try:
        return datetime.strptime(
            date_part + time_part + millis,
            "%Y%m%d%H%M%S%f"
        )

    except ValueError:
        return None


# ============================================================
# FUZZY CHARACTER SIMILARITY
# ============================================================

def sequence_similarity(a, b):
    """
    Calculate position-by-position similarity between two strings.

    Example:

        BLOCKED
        DLOCKED

    Most characters match, so similarity is high.

    This is intentionally strict to avoid false positives.
    """

    if not a or not b:
        return 0.0

    length = min(len(a), len(b))

    if length == 0:
        return 0.0

    matches = sum(
        char_a == char_b
        for char_a, char_b in zip(a[:length], b[:length])
    )

    # Penalize different lengths.
    return matches / max(len(a), len(b))


# ============================================================
# BEST TARGET OVERLAP
# ============================================================

def best_target_match(decoded_text):
    """
    Find the best alignment of the decoded text against the
    target message.

    Returns:

        similarity, target_position

    Example:

        Target:
            AIRFILTERISBLOCKED

        Decoded:
            ERISDLOCKED

    The function compares the decoded text with every possible
    same-length section of the target.
    """

    text = normalize_text(decoded_text)
    target = NORMALIZED_TARGET

    if not text or not target:
        return 0.0, -1

    # Exact target.
    if text == target:
        return 1.0, 0

    best_score = 0.0
    best_position = -1

    # --------------------------------------------------------
    # Compare decoded text against every possible target
    # substring of the same length.
    # --------------------------------------------------------

    if len(text) <= len(target):

        for start in range(
            0,
            len(target) - len(text) + 1
        ):

            target_part = target[
                start:start + len(text)
            ]

            score = sequence_similarity(
                text,
                target_part
            )

            if score > best_score:

                best_score = score
                best_position = start

    else:

        # Decoded text is longer than target.
        score = sequence_similarity(
            text[:len(target)],
            target
        )

        best_score = score
        best_position = 0

    return best_score, best_position


# ============================================================
# EXACT TARGET FRAGMENT
# ============================================================

def is_exact_target_fragment(decoded_text):
    """
    True when the decoded text is an exact contiguous fragment
    of the target.

    This handles scrolling fragments such as:

        A
        AI
        AIR
        AIR FILTER
        IR FILTER IS
        FILTER IS BL
        ER IS BLOCKED
    """

    text = normalize_text(decoded_text)

    if not text:
        return False

    return text in NORMALIZED_TARGET


# ============================================================
# STRONG TARGET FRAME
# ============================================================

def is_strong_target_frame(decoded_text):
    """
    Determine whether a frame provides strong evidence that it
    belongs to the target scrolling message.

    Rules:

    1. Exact target substring -> accepted.
    2. Longer meaningful fragment with >= 90% positional
       similarity -> accepted.

    The 90% threshold is important.

    Example:

        ERISDLOCKED
        ERISBLOCKED

    similarity is high enough.

    But:

        ERISDLOCKEB

    has an additional incorrect final character and falls below
    the threshold.
    """

    text = normalize_text(decoded_text)

    if not text:
        return False

    # Exact substring.
    if is_exact_target_fragment(text):
        return True

    # Very short strings should not independently establish
    # a target run.
    if len(text) < 4:
        return False

    score, _ = best_target_match(text)

    # Strict threshold to prevent unrelated frames.
    return score >= 0.90


# ============================================================
# FIND STRONG TARGET FRAMES
# ============================================================

def find_strong_target_frames(decoded_frames):
    """
    Find frames with strong target evidence.

    Short fragments such as A and AI are intentionally recovered
    later from the surrounding scrolling sequence.
    """

    matches = []

    for item in decoded_frames:

        if is_strong_target_frame(item["text"]):
            matches.append(item)

    return matches


# ============================================================
# GROUP STRONG TARGET FRAMES
# ============================================================

def group_target_frames(matches, max_gap=3):
    """
    Group strong target frames.

    Small gaps are tolerated because a decoder can occasionally
    fail on one frame.
    """

    if not matches:
        return []

    groups = []

    current_group = [matches[0]]

    for current in matches[1:]:

        previous = current_group[-1]

        gap = (
            current["frame"]
            - previous["frame"]
        )

        if gap <= max_gap:

            current_group.append(current)

        else:

            groups.append(current_group)

            current_group = [current]

    groups.append(current_group)

    return groups


# ============================================================
# CHECK SCROLLING PROGRESSION
# ============================================================

def can_belong_to_same_scroll(
    previous_text,
    current_text
):
    """
    Determine whether two neighboring decoded frames can belong
    to the same scrolling message.

    This is used to recover short beginning fragments such as:

        A
        AI
        AIR

    when the strong matcher begins at AIR.
    """

    previous = normalize_text(previous_text)
    current = normalize_text(current_text)

    if not previous or not current:
        return False

    # --------------------------------------------------------
    # If either is an exact target fragment, this is strong
    # evidence.
    # --------------------------------------------------------

    if previous in NORMALIZED_TARGET:
        return True

    if current in NORMALIZED_TARGET:
        return True

    # --------------------------------------------------------
    # Compare current frame with target.
    # --------------------------------------------------------

    current_score, _ = best_target_match(current)

    previous_score, _ = best_target_match(previous)

    if (
        current_score >= 0.90
        or previous_score >= 0.90
    ):
        return True

    # --------------------------------------------------------
    # Check whether the current frame extends the previous
    # scrolling sequence.
    #
    # Example:
    #
    # A
    # AI
    # AIR
    #
    # or:
    #
    # AIR FILTER IS
    # IR FILTER IS
    # R FILTER IS D
    # --------------------------------------------------------

    common_prefix = 0

    for a, b in zip(previous, current):

        if a == b:
            common_prefix += 1
        else:
            break

    if common_prefix >= 2:
        return True

    # Check overlap between end of previous and beginning of
    # current.
    max_overlap = min(
        len(previous),
        len(current)
    )

    for size in range(
        max_overlap,
        1,
        -1
    ):

        if previous[-size:] == current[:size]:
            return True

    return False


# ============================================================
# EXPAND TARGET RUN
# ============================================================

def expand_target_run(
    decoded_frames,
    strong_run,
    max_backward=10,
    max_forward=10
):
    """
    Expand a strong target run to include short scrolling
    fragments at the beginning and valid fragments at the end.
    """

    if not strong_run:
        return []

    frame_to_index = {
        item["frame"]: index
        for index, item in enumerate(decoded_frames)
    }

    first_frame = strong_run[0]["frame"]
    last_frame = strong_run[-1]["frame"]

    first_index = frame_to_index[first_frame]
    last_index = frame_to_index[last_frame]

    selected_indices = set(
        range(
            first_index,
            last_index + 1
        )
    )

    # ========================================================
    # BACKWARD
    # ========================================================

    current_index = first_index

    for _ in range(max_backward):

        previous_index = current_index - 1

        if previous_index < 0:
            break

        previous = decoded_frames[
            previous_index
        ]

        current = decoded_frames[
            current_index
        ]

        if not previous["text"].strip():
            break

        if can_belong_to_same_scroll(
            previous["text"],
            current["text"]
        ):

            selected_indices.add(
                previous_index
            )

            current_index = previous_index

        else:
            break

    # ========================================================
    # FORWARD
    # ========================================================

    current_index = last_index

    for _ in range(max_forward):

        next_index = current_index + 1

        if next_index >= len(decoded_frames):
            break

        current = decoded_frames[
            current_index
        ]

        following = decoded_frames[
            next_index
        ]

        if not following["text"].strip():
            break

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Do NOT blindly accept a fuzzy frame here.
        #
        # The next frame must either:
        #
        #   1. be an exact target fragment, or
        #   2. have >= 90% similarity.
        #
        # This prevents:
        #
        # ER IS DLOCKEB
        #
        # from being accepted after:
        #
        # ER IS DLOCKED
        # ----------------------------------------------------

        if is_exact_target_fragment(
            following["text"]
        ):

            selected_indices.add(
                next_index
            )

            current_index = next_index

            continue

        score, _ = best_target_match(
            following["text"]
        )

        if score >= 0.90:

            selected_indices.add(
                next_index
            )

            current_index = next_index

            continue

        break

    return [
        decoded_frames[index]
        for index in sorted(selected_indices)
    ]


# ============================================================
# SELECT STRONGEST TARGET RUN
# ============================================================

def select_target_run(groups):
    """
    Select the strongest continuous target run.
    """

    if not groups:
        return None

    return max(
        groups,
        key=lambda group: (
            len(group),
            group[-1]["frame"]
            - group[0]["frame"]
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("METHOD 2 - TEMPLATE MATCHING EVALUATION")
    print("=" * 80)

    print()

    print(
        f"Target message           : '{TARGET}'"
    )

    # ========================================================
    # LOAD TEMPLATES
    # ========================================================

    templates = load_templates()

    if not templates:

        print()
        print("ERROR: No templates found.")
        print()

        return

    # ========================================================
    # LOAD BMP FILES
    # ========================================================

    image_paths = sorted(
        DATA_DIR.glob("*.bmp")
    )

    if not image_paths:

        print()
        print(
            "ERROR: No BMP images found in:"
        )
        print(DATA_DIR)
        print()

        return

    # ========================================================
    # FRAME INFORMATION
    # ========================================================

    timestamped_paths = []

    for index, image_path in enumerate(
        image_paths
    ):

        timestamp = timestamp_from_filename(
            image_path
        )

        timestamped_paths.append(
            {
                "path": image_path,
                "frame": index + 1,
                "timestamp": timestamp
            }
        )

    # ========================================================
    # AUTOMATIC MULTI-DATASET CALIBRATION
    # ========================================================
    #
    # Suppress calibration diagnostics so that the evaluator
    # prints only the final result.
    #
    # IMPORTANT:
    # calibrate_multiple_cell_starts() receives FILE PATHS.
    # ========================================================

    calibration_output = io.StringIO()

    with redirect_stdout(
        calibration_output
    ), redirect_stderr(
        calibration_output
    ):

        calibration = calibrate_multiple_cell_starts(
            [
                item["path"]
                for item in timestamped_paths
            ]
        )

    frame_groups = calibration[
        "frame_groups"
    ]

    calibrations = calibration[
        "calibrations"
    ]

    # ========================================================
    # DECODE ALL FRAMES
    # ========================================================

    decoded_frames = []

    for index, item in enumerate(
        timestamped_paths
    ):

        image = cv2.imread(
            str(item["path"])
        )

        if image is None:
            continue

        # ----------------------------------------------------
        # Calibration for this frame
        # ----------------------------------------------------

        if index < len(frame_groups):

            group_id = frame_groups[
                index
            ]

            cell_starts = calibrations[
                group_id
            ]

        else:

            cell_starts = None

        # ----------------------------------------------------
        # Template decoding
        # ----------------------------------------------------

        decoded_text, confidences = decode_frame(
            image,
            templates,
            cell_starts
        )

        decoded_frames.append(
            {
                "frame": item["frame"],
                "timestamp": item["timestamp"],
                "text": decoded_text.strip()
            }
        )

    # ========================================================
    # FIND STRONG TARGET FRAMES
    # ========================================================

    matches = find_strong_target_frames(
        decoded_frames
    )

    if not matches:

        print()
        print("=" * 80)
        print("TARGET MESSAGE NOT FOUND")
        print("=" * 80)
        print()

        return

    # ========================================================
    # GROUP STRONG MATCHES
    # ========================================================

    groups = group_target_frames(
        matches,
        max_gap=3
    )

    strong_run = select_target_run(
        groups
    )

    if not strong_run:

        print()
        print(
            "Target run could not be determined."
        )
        print()

        return

    # ========================================================
    # EXPAND RUN
    # ========================================================
    #
    # This recovers short beginning fragments such as:
    #
    # A
    # AI
    #
    # when the strong run begins at:
    #
    # AIR
    #
    # It also preserves the last valid target frame.
    # ========================================================

    target_run = expand_target_run(
        decoded_frames,
        strong_run,
        max_backward=10,
        max_forward=10
    )

    if not target_run:

        print()
        print(
            "Target message could not be established."
        )
        print()

        return

    # ========================================================
    # FIRST / LAST MATCHING FRAME
    # ========================================================

    first_target = target_run[0]

    last_target = target_run[-1]

    start_time = first_target[
        "timestamp"
    ]

    end_time = last_target[
        "timestamp"
    ]

    # ========================================================
    # DISPLAY DURATION
    # ========================================================

    if (
        start_time is not None
        and end_time is not None
    ):

        duration = (
            end_time - start_time
        ).total_seconds()

    else:

        duration = None

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()

    print("=" * 80)
    print("TARGET MESSAGE DISPLAY TIME")
    print("=" * 80)

    print()

    print(
        f"Message                  : '{TARGET}'"
    )

    print(
        f"First matching frame     : "
        f"{first_target['frame']}"
    )

    if start_time is not None:

        print(
            f"First matching timestamp : "
            f"{start_time.strftime('%H:%M:%S.%f')[:-3]}"
        )

    else:

        print(
            "First matching timestamp : unavailable"
        )

    print(
        f"Last matching frame      : "
        f"{last_target['frame']}"
    )

    if end_time is not None:

        print(
            f"Last matching timestamp  : "
            f"{end_time.strftime('%H:%M:%S.%f')[:-3]}"
        )

    else:

        print(
            "Last matching timestamp  : unavailable"
        )

    if duration is not None:

        print(
            f"Display duration         : "
            f"{duration:.3f} seconds"
        )

    else:

        print(
            "Display duration         : unavailable"
        )

    print()

    print("=" * 80)
    print("METHOD 2 EVALUATION COMPLETE")
    print("=" * 80)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()