# Chest X-ray Detection 🫁

This project leverages deep learning to classify chest X-ray images into various categories, such as normal lungs and different lung diseases. The system is composed of a **FastAPI backend** for inference and a **Streamlit frontend** for user interaction.

---

## Features

- **Deep Learning Model**: Utilizes a YOLO-based model for image classification.
- **FastAPI Backend**: Handles file uploads and performs image classification.
- **Streamlit Frontend**: Allows users to upload images, view predictions, and inspect probabilities in an intuitive interface.
- **Responsive Design**: Displays results in real-time with probability scores and image previews.

---

## Project Structure

### Backend: `app.py`

The backend is powered by **FastAPI**:
- **Endpoint**: `/Chest_Xray_Detection`
- **Input**: A chest X-ray image (`jpg`, `jpeg`, or `png` format).
- **Output**: JSON containing:
  - Predicted class.
  - Probability score (percentage).

### Frontend: `main.py`

The frontend is built with **Streamlit**:
- **File Upload**: Allows users to upload X-ray images.
- **Prediction Display**: Fetches predictions from the backend and displays them.
- **Image Viewer**: Shows the uploaded image alongside predictions.

---

## Dataset

The model was trained on a carefully curated dataset of chest X-ray images. The dataset includes images from multiple public datasets, with rigorous cleaning to ensure quality and uniqueness.

### Classes

1. **Normal Lungs**: Healthy lung images.
2. **Diseased Lungs**:
   - Bacterial Pneumonia
   - Coronavirus Disease (COVID-19)
   - Tuberculosis
   - Viral Pneumonia

### Data Augmentation

To improve performance and generalization, data augmentation techniques were applied:
- Rotations
- Flipping (horizontal/vertical)
- Contrast adjustments

The augmented dataset consists of approximately **10,000 images**, divided into:
- **Training Set**: For model training.
- **Validation Set**: For hyperparameter tuning.
- **Test Set**: For evaluating model performance.

---

## Setup and Installation

### Prerequisites

- Python 3.8+
- Libraries: `fastapi`, `streamlit`, `ultralytics`, `PIL`, `numpy`, `requests`, `uvicorn`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Osama-Abo-Bakr/Chest_X-ray_Detection.git
   cd Chest_X-ray_Detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place the YOLO model file (`model_yolo-v1.pt`) in the `models/` directory.

### Run the Project

1. Start the backend server:
   ```bash
   uvicorn app:app --reload
   ```

2. Run the frontend:
   ```bash
   streamlit run main.py
   ```

3. Open the Streamlit app in your browser:
   ```
   http://localhost:8501
   ```

---

## How to Use

1. Upload a chest X-ray image via the Streamlit app.
2. Wait for the prediction.
3. View the predicted class and probability score.

---

## Future Enhancements

- Support for additional diseases.
- Improved model performance with larger datasets.
- Integration with cloud storage for dataset management.

---

## Acknowledgements

Special thanks to public datasets and the open-source community for providing resources to build this project.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.