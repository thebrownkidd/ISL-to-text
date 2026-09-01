# ISL-to-Text

An attention-based approach to convert Indian Sign Language (ISL) to text using simulated hand gesture data. This repository contains the ISL-to-text component of a larger bilingual generation system.

## Context

This work is part of **Annuvadak: An Automated Bilingual Generation System**, a research project funded by SERB (Science and Engineering Research Board, 2022-2025) and supervised by **Rupali Bhardwaj** at Thapar Institute of Engineering and Technology. The ISL-to-text pipeline addresses the challenge of limited training data for Indian Sign Language recognition by simulating new gesture samples from existing data.

## Approach

### Data Simulation

The system tackles ISL data scarcity through a landmark-based augmentation method:

1. Extract 21 hand landmark points (P0-P20) and 22 pose keypoints from video frames using MediaPipe
2. Decompose hand movements into **spatial translations** (hand position in 3D space) and **gestural translations** (changes in the sign itself)
3. Apply computed transformation matrices to base examples to generate synthetic training samples
4. Post-process generated data with centralization and rotation correction

This provides a low-complexity alternative to fine-tuning large pretrained models.

### Model Architecture

A **Transformer encoder** processes the landmark data:

- **Embedding layers**: Hand landmarks (42-dim) and pose landmarks (44-dim) are projected to 126-dimensional embeddings via Tanh-activated dense layers
- **Multi-head attention**: Single-head attention (embed_dim=126) with query/key/value projections
- **Transformer blocks**: Attention + layer normalization + feed-forward (126 to 504 to 126) + residual connections + dropout (0.2)
- **Output**: Conversion layer maps to 114 classes via softmax

The decoder component is defined but not active in the current version; the model operates as an encoder-only classifier.

### Results

The model achieves approximately **66.6% accuracy** on the ISL-to-text classification task. This represents the isolated ISL-to-text subset of the broader Annuvadak system.

## Repository Structure

```
.
├── 0-9/                        # Numerical gesture data
├── INCLUDE 50/                 # Dataset subset
├── Literature/                 # Reference papers
├── Testing/
│   └── Test.py                 # Evaluation script
├── Traning/
│   ├── architecturemodel.ipynb # Model architecture notebook
│   ├── Traningloop.ipynb       # Training loop
│   ├── Perspective transf.ipynb# Perspective transformation
│   ├── test.ipynb              # Testing notebook
│   ├── model_arc.py            # Model architecture (PyTorch)
│   └── model.pt                # Trained model weights
├── Word Level ISL/
│   ├── Model.ipynb             # Word-level model
│   ├── Perspective transf.ipynb# Word-level perspective transforms
│   ├── model.pt                # Word-level trained weights
│   └── test.ipynb              # Word-level testing
├── face_landmarker.task        # MediaPipe face landmark model
├── Data License
├── License                     # Apache-2.0
└── README.md
```

## Dependencies

- Python 3.x
- PyTorch
- MediaPipe (PoseLandmarker, HandLandmarker)
- NumPy
- OpenCV

## Usage

1. Place input video frames or gesture data in the appropriate data directory
2. Run the training notebook (`Traning/Traningloop.ipynb`) to train the model, or load the pretrained weights from `model.pt`
3. Evaluate using `Testing/Test.py` or the test notebooks

## Acknowledgments

- **Funding**: Science and Engineering Research Board (SERB), Government of India (project funded 2022-2025)
- **Supervision**: Rupali Bhardwaj, Thapar Institute of Engineering and Technology, Patiala
- **Institutional support**: Thapar Institute of Engineering and Technology

## License

Apache-2.0. See `License` file for details.
