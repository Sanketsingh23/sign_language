# Real-Time Sign Language Recognition

Real-time webcam-based gesture recognition with MediaPipe hand landmarks and two trained models:

- a Random Forest classifier for word gestures such as `YES`, `NO`, `OK`, `HELLO`, and `STOP`
- an LSTM classifier for alphabet gestures `A` to `Z`

The project includes three interfaces:

- `final_app.py`: OpenCV desktop app that switches between word and alphabet modes
- `streamlit_app.py`: Streamlit web app with the same model pipeline
- `tkinter_app.py`: legacy Tkinter demo built around the LSTM model

## Features

- Live webcam capture and hand landmark detection with MediaPipe
- Normalized 21-point hand keypoints for each frame
- Word-mode prediction with a trained Random Forest model
- Alphabet-mode prediction with a trained LSTM sequence model
- Sentence buffering with simple stability logic to avoid repeated duplicates
- Multiple front ends for desktop and browser-based usage

## Project Layout

- `dataset/`: stored `.npy` samples for the word model
- `gesture_model.pkl`: pretrained Random Forest model for word gestures
- `model.h5` and `model.json`: pretrained alphabet model
- `collect_data.py`: capture normalized keypoint samples into `dataset/`
- `train_model.py`: train the Random Forest word model from the current flat dataset
- `data.py`: legacy sequence export script for the older `MP_Data` workflow
- `trainmodel.py`: legacy LSTM training script for sequence-based data
- `function.py`: shared MediaPipe helpers and keypoint utilities

## Requirements

- Python 3.9 or 3.10 is recommended
- A working webcam
- Windows, macOS, or Linux with OpenCV camera access

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

## Setup

1. Clone the repository and open the project folder.
2. Create and activate a virtual environment.
3. Install dependencies with `pip install -r requirements.txt`.
4. Make sure the pretrained files are present in the project root:
   - `gesture_model.pkl`
   - `model.h5`
   - `model.json`

## Run the apps

### OpenCV desktop app

```bash
python final_app.py
```

Controls inside the app:

- `1`: switch to word mode
- `2`: switch to alphabet mode
- `c`: clear the current sentence
- `q`: quit

### Streamlit app

```bash
streamlit run streamlit_app.py
```

Use the sidebar to start/stop capture, switch between `WORD` and `ALPHABET`, and clear the sentence.

### Tkinter app

```bash
python tkinter_app.py
```

This is an older demo interface that uses the alphabet model and a Tkinter window.

### Camera test

```bash
python camera_test.py
```

Use this if you only want to verify that the webcam is opening correctly.

## Data Format

The current word dataset uses normalized hand keypoints saved as `.npy` files in class folders:

```text
dataset/
  YES/
  NO/
  OK/
  STOP/
  HELLO/
```

Each sample is a flattened vector of 63 values, representing 21 hand landmarks with `(x, y, z)` coordinates.

## Training

### Word model

Train the Random Forest model from the current dataset with:

```bash
python train_model.py
```

This reads the `.npy` files from `dataset/` and writes `gesture_model.pkl`.

### Alphabet model

The alphabet workflow is based on 15-frame sequences and the pretrained files `model.json` and `model.h5`.

The scripts `data.py` and `trainmodel.py` are legacy helpers for the older sequence-based pipeline. They expect an `MP_Data/` directory structure and are not used by the current flat `dataset/` workflow.

## Collecting New Word Samples

Use `collect_data.py` to capture normalized keypoint samples for the word model.

The script saves samples into `dataset/<label>/` as `.npy` files. Update the `actions` list in the script if you want to collect different labels.

## How It Works

1. MediaPipe detects one hand and extracts 21 landmarks.
2. The keypoints are normalized relative to the first landmark.
3. Word mode sends the 63-value vector to the Random Forest classifier.
4. Alphabet mode builds a 15-frame sequence and sends it to the LSTM model.
5. The UI keeps a sentence buffer and filters repeated predictions using a delay-based stability check.

## Notes

- The repository includes both current scripts and older experimental scripts; the README focuses on the working flow used by `final_app.py`, `streamlit_app.py`, and `train_model.py`.
- If you retrain the models, keep the class order and input shapes consistent with the existing code.
- Camera permissions may need to be enabled at the OS level before the apps can read from the webcam.

## Troubleshooting

- If the camera does not open, close other apps that may be using it and try `camera_test.py` first.
- If a model file is missing, confirm that `gesture_model.pkl`, `model.h5`, and `model.json` are in the project root.
- If Streamlit fails to start, verify that `streamlit` is installed in the active environment.

## License

No license file is included in the repository. Add one before publishing or redistributing the project.
