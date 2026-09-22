# HIL Test Automation – Task 2: 16-Segment Display Text Detection

## Overview

This project implements a computer vision-based solution for detecting and decoding text displayed on a **16-segment display**.

The input consists of a sequence of image frames containing text displayed on the segmented display. The system processes the images and converts the illuminated segments into readable characters.

The project implements multiple approaches for text detection and also determines the time duration for which a specific message is displayed.

**Target Message**

```
AIR FILTER IS BLOCKED
```

---

## Task Description

The task consists of detecting text from images of a 16-segment display.

The requirements are:

1. Identify at least three possible methods for accurately detecting the displayed text.
2. Implement at least two methods using custom code.
3. Process the given image sequence.
4. Decode the characters displayed on the 16-segment display.
5. Determine when the target message appears and calculate its display duration.

---

## Implemented Methods

### Method 1 – 16-Segment Pattern Recognition

The first method uses the structure of the 16-segment display directly.

Each character is represented using a predefined combination of illuminated segments, for example:

```
A → specific segment combination
B → specific segment combination
C → specific segment combination
...
Z → specific segment combination
```

**The system:**

1. Loads an input image.
2. Detects the illuminated display region.
3. Automatically determines the character-cell positions.
4. Divides the display into individual character cells.
5. Samples the predefined 16 segment locations.
6. Determines whether each segment is ON or OFF.
7. Creates a binary segment pattern.
8. Compares the pattern with predefined character patterns.
9. Selects the character with the closest matching pattern.
10. Combines the detected characters to produce the complete text.

**Advantages**

- Does not require an external OCR API
- Works directly with the known 16-segment hardware
- Fast and lightweight
- Can work with characters that are not present in the training images, if their segment pattern is defined

---

### Method 2 – Template Matching

The second method uses image templates generated from the input dataset.

The template-building process first uses Method 1 to identify characters and extracts good-quality examples of those characters.

The extracted characters (for the corresponding dataset) are:

```
C  D  E  G  I  N  O  S  U
```

The templates are stored inside `outputs/templates/`:

```
outputs/
└── templates/
    ├── C.png
    ├── D.png
    ├── E.png
    ├── G.png
    ├── I.png
    ├── N.png
    ├── O.png
    ├── S.png
    └── U.png
```

**For every new character cell:**

1. The character image is extracted.
2. The image is normalized.
3. The character is resized to a common size.
4. It is compared against all available templates.
5. A similarity score is calculated.
6. The template with the highest similarity is selected.

**Advantages**

- Simple and fast
- Does not require a machine-learning model
- Templates can be rebuilt automatically for a new dataset
- Useful when the display appearance remains consistent

---

### Method 3 – OCR-Based Detection

A third possible approach is to use an OCR system such as:

- Tesseract OCR
- EasyOCR
- PaddleOCR

The image would first be preprocessed and then passed to the OCR engine.

> Standard OCR is **not** the primary approach in this project because the input is a 16-segment display, where direct segment-based recognition provides more control over the character structure.

---

## System Architecture

```
Input BMP Images
       |
       v
Image Preprocessing
       |
       v
Display Mask Detection
       |
       v
Automatic Display Calibration
       |
       v
Character Cell Detection
       |
       +----------------------+
       |                      |
       v                      v
   Method 1               Method 2
16-Segment              Template
Recognition             Matching
       |                      |
       v                      v
Decoded Characters    Similarity Matching
       |                      |
       +----------+-----------+
                   |
                   v
             Detected Text
                   |
                   v
          Target Message Check
                   |
                   v
           Display Duration
```

---

## Automatic Calibration

One of the important parts of the implementation is **automatic calibration**.

The original image dataset used fixed character positions. However, this approach is not reliable when a new dataset has a different horizontal alignment. Therefore, the project automatically detects the character positions from the images.

**The calibration process:**

1. Creates a display mask.
2. Calculates the number of active pixels for each image column.
3. Detects active regions.
4. Finds character centers.
5. Estimates the distance between neighboring character centers.
6. Calculates the display pitch.
7. Estimates the horizontal phase.
8. Generates character-cell start positions.

**Example calibration output:**

```
AUTOMATIC DISPLAY CALIBRATION

Frames inspected : 30
Vertical samples : 414
Estimated pitch  : 61.95 px
Estimated phase  : 2.06 px

Detected cells   : 13

Cell starts:
[2.06, 64.02, 125.97, 187.92, 249.87,
 311.81, 373.77, 435.72, 497.67,
 559.62, 621.57, 683.52, 745.47]
```

### Multi-Dataset Calibration

The project was further improved to handle datasets where different groups of frames have different horizontal alignment.

Instead of assuming that every frame has the same calibration, the system:

1. Estimates the display phase for each frame.
2. Groups frames having similar calibration.
3. Creates calibration values for each group.
4. Uses the appropriate calibration for each frame.

This prevents the decoder from depending on a specific frame number or hardcoded dataset boundary. The system therefore **does not** use logic such as:

```python
if frame_number <= 100:
    ...
else:
    ...
```

Instead, calibration is determined from the image data itself.

---

## Project Structure

```
HIL-Test-Automation/
│
├── data/
│   └── text/
│       ├── frame_001.bmp
│       ├── frame_002.bmp
│       ├── ...
│       └── frame_N.bmp
│
├── outputs/
│   ├── templates/
│   │   ├── A.png
│   │   ├── B.png
│   │   └── ...
│   │
│   └── text/
│       ├── preprocessed_first_frame.png
│       └── segment_decoder_results.csv
│
├── src/
│   ├── __init__.py
│   ├── segment_decoder.py
│   ├── build_character_templates.py
│   ├── template_decoder.py
│   └── evaluate_template_decoder.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Technologies Used

| Category | Details |
|---|---|
| **Programming Language** | Python |
| **Computer Vision** | OpenCV, NumPy |
| **Image Processing** | HSV color space, thresholding, binary masks, contour/region analysis, image normalization, template matching |
| **Data Processing** | CSV, file/directory processing, timestamp extraction |

**Main libraries used:**

```
opencv-python
numpy
```

**Additional Python standard libraries:**

```
pathlib
re
math
csv
dataclasses
collections
```

---

## Method 1 Workflow

```
BMP Image
   |
   v
HSV Conversion
   |
   v
Display Mask
   |
   v
Automatic Calibration
   |
   v
Character Cells
   |
   v
16-Segment Sampling
   |
   v
ON/OFF Segment Pattern
   |
   v
Pattern Comparison
   |
   v
Character
```

For each character cell, the system calculates how much of each segment is illuminated. The segment is classified as **ON** or **OFF**. The resulting 16-bit pattern is compared with the predefined segment patterns for supported characters.

---

## Method 2 Workflow

```
Input Image
     |
     v
Display Mask
     |
     v
Character Cell Extraction
     |
     v
Character Normalization
     |
     v
Resize
     |
     v
Compare With Templates
     |
     v
Similarity Score
     |
     v
Best Matching Character
```

The character templates are automatically generated using detected characters from the dataset.

### Character Template Generation

Templates are generated using `build_character_templates.py`.

**The script:**

1. Loads the input images.
2. Performs automatic calibration.
3. Decodes frames using Method 1.
4. Identifies characters.
5. Extracts character images.
6. Normalizes the characters.
7. Evaluates candidate quality.
8. Selects high-quality candidates.
9. Saves them as PNG templates.

**Run:**

```bash
python src/build_character_templates.py
```

---

## Running the Project

### Running Method 1

```bash
python src/segment_decoder.py
```

The decoder processes all BMP files inside `data/text/` and prints the detected text for each frame.

**Example:**

```
Frame 001: ENGINE ECU
Frame 019: ENGINE ECU I
Frame 035: ENGINE ECU IS
...
```

### Running Method 2

After generating templates, run:

```bash
python src/template_decoder.py
```

The program loads the generated templates and performs template-based character recognition.

### Evaluating the Target Message

The target message is:

```
AIR FILTER IS BLOCKED
```

The evaluation script checks the decoded frames and identifies the period during which the target message is displayed.

**Run:**

```bash
python src/evaluate_template_decoder.py
```

**The script:**

1. Reads decoded frames.
2. Extracts timestamps from filenames.
3. Normalizes the detected text.
4. Searches for the target message or its valid fragments.
5. Finds the first target frame.
6. Finds the last target frame.
7. Calculates the time difference.

---

## Display Duration Calculation

The duration is calculated from the timestamps in the image filenames.

| Event | Timestamp |
|---|---|
| First detected frame | 09:59:24.661 |
| Last detected frame | 09:59:33.523 |

**Observed duration: 8.862 seconds**

```
Duration = Last Timestamp − First Timestamp
```

---

## Handling New Datasets

The implementation was designed to avoid depending on a specific dataset size or fixed frame positions.

**The system dynamically:**

- Finds BMP files
- Detects the display area
- Estimates character-cell positions
- Performs calibration
- Groups frames with different horizontal phases
- Builds character templates from the current dataset
- Detects characters using segment patterns

Therefore, a new dataset can be placed inside `data/text/` and processed **without** changing the frame count or hardcoding frame numbers.

---

## What Is Configured in the Code?

The project does not hardcode the actual message or frame boundaries.

However, some configuration is intentionally predefined because the system is designed for a known 16-segment display hardware. These include:

- 16-segment geometry
- Segment sampling locations
- Character-to-segment mappings
- Basic cell dimensions
- Image preprocessing thresholds
- Segment detection thresholds

This is different from hardcoding the dataset. For example, the system does **not** assume:

```
Frame 1–100   = Dataset 1
Frame 101–205 = Dataset 2
```

Instead, calibration is calculated from the image data.

---

## Troubleshooting a New Dataset

If a new dataset produces incorrect results, follow this debugging process:

### 1. Check Method 1 first

```bash
python src/segment_decoder.py
```

If Method 1 is incorrect, inspect:

- Calibration
- Display mask
- Character cell positions
- Segment geometry
- Character segment mappings
- Thresholds

### 2. Check calibration

Look at:

- Estimated pitch
- Estimated phase
- Detected cells
- Cell starts

If these values are incorrect, the character cells are probably being extracted from the wrong locations.

### 3. Check character mapping

If the segments are detected correctly but the character is wrong, verify the character's 16-segment pattern in `segment_decoder.py`.

### 4. Rebuild templates

If Method 1 is correct but Method 2 is incorrect:

```bash
python src/build_character_templates.py
python src/template_decoder.py
```

### 5. Check templates

Inspect `outputs/templates/`. The generated templates should contain clean representations of the detected characters.

### 6. Check preprocessing

If the new dataset has different brightness, color, or contrast, HSV thresholding may need adjustment.

---

## Challenges Faced

### 1. Different Horizontal Alignment

The biggest challenge was that different datasets could have different horizontal positions of the character cells. A fixed configuration such as `CELL_STARTS = [...]` worked for one dataset but failed for another.

**Solution:** Automatic calibration was implemented to estimate character pitch, horizontal phase, and character-cell positions from the image data.

### 2. Different Dataset Groups

When multiple datasets were combined, one global calibration could work for the first dataset but fail for another.

**Solution:** Multi-calibration was implemented. The system estimates the phase for individual frames and groups frames with similar calibration.

### 3. Character Recognition

A 16-segment display does not behave like normal printed text. Characters can have similar segment patterns, making recognition sensitive to segment position, thresholds, image quality, and display brightness.

**Solution:** The segment-based method uses predefined 16-segment character patterns and similarity-based matching.

### 4. Template Quality

Template matching depends heavily on the quality of the generated templates. Poorly extracted characters can produce incorrect template matches.

**Solution:** The template builder evaluates candidate characters and selects higher-quality samples before saving the templates.

### 5. Lighting and Image Variations

Changes in brightness or display intensity can affect the binary display mask.

**Solution:** HSV-based preprocessing and segment illumination thresholds are used to separate illuminated display pixels from the background.

---

## Limitations

This project is designed specifically for a known 16-segment display structure. It is **not** a general OCR system.

Performance can decrease when:

- The display hardware changes
- Segment geometry changes significantly
- Image resolution changes significantly
- The display is heavily distorted
- Lighting conditions are very different
- Characters outside the supported segment mapping are introduced

For a completely different display type, the segment geometry and character mappings would need to be adapted.

---

## Possible Improvements

1. **Perspective Correction** — Use homography/perspective transformation if the display is viewed at an angle.
2. **Automatic Segment Geometry Detection** — Instead of predefined segment coordinates, detect the segment positions automatically.
3. **Adaptive Thresholding** — Use adaptive thresholds based on the brightness of each individual frame.
4. **Temporal Smoothing** — Use information from neighboring frames to reduce character fluctuations. For example:

   ```
   Frame 10 → ENGINE
   Frame 11 → ENG1NE
   Frame 12 → ENGINE
   ```

   Temporal voting could select `ENGINE` as the final result.

5. **Machine Learning** — A CNN or lightweight image classification model could be trained to classify individual 16-segment characters.
6. **OCR Comparison** — The custom methods could be compared against Tesseract, EasyOCR, and PaddleOCR to evaluate their accuracy on the same dataset.

---

## Why Custom Computer Vision Was Used

The project uses custom computer vision instead of relying entirely on external OCR APIs because the input is a structured 16-segment display.

The display provides useful prior information:

```
16 known segments + known character patterns = direct character recognition
```

This makes the solution lightweight and explainable.

---

## Output

The system produces decoded text for each frame. It can also generate:

- `outputs/templates/` — character templates
- `outputs/text/` — processing results

The output can be used to determine:

- Detected characters
- Detected messages
- Frame ranges
- Target message occurrence
- Display duration

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/IshankSingla/HIL-Test-Automation.git
```

**2. Move into the project**

```bash
cd HIL-Test-Automation
```

**3. Create a virtual environment**

```bash
python -m venv venv
```

**4. Activate it (Windows)**

```bash
venv\Scripts\activate
```

**5. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Usage

Place the input BMP images inside `data/text/`, then run:

```bash
# Method 1
python src/segment_decoder.py

# Generate templates
python src/build_character_templates.py

# Method 2
python src/template_decoder.py

# Evaluate the target message
python src/evaluate_template_decoder.py
```

---

## Requirements

The project requires Python and the libraries specified in `requirements.txt`.

**Main dependencies:**

```
opencv-python
numpy
```

---

## Key Features

- 16-segment display recognition
- Custom computer vision implementation
- Automatic character-cell calibration
- Multi-dataset calibration
- Dynamic frame discovery
- Automatic character template generation
- Template-based character recognition
- Timestamp-based duration calculation
- No external OCR API required
- No paid API key required
- Designed for new datasets with the same display hardware

---

## Conclusion

This project demonstrates a computer vision-based approach for extracting text from a 16-segment display.

Two custom methods were implemented:

1. **16-Segment Pattern Recognition**
2. **Template Matching**

A third possible approach is OCR-based recognition.

The project also includes automatic calibration so that the solution is not dependent on fixed frame numbers or a single dataset alignment. The main focus of the implementation is to make the solution adaptable to different image sequences while maintaining the known geometry and characteristics of the 16-segment display.
