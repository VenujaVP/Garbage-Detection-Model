# Garbage Detection Using YOLOv8

A comprehensive real-time garbage detection system built with YOLOv8 deep learning framework for automated waste identification and classification.

## 🗂️ Project Structure

```
@campus/
├── Garbage Detection Model Traning Process.ipynb    # Model training notebook (Colab)
├── real time detection and test video.ipynb        # Local inference & testing
├── video.mp4                                        # Sample input video
├── output_video_with_results.mp4                   # Processed output video
├── dataset.yaml                                     # Dataset configuration file
├── best1.pt                                         # Best trained model
├── model1.pt                                        # Alternative trained model
└── ReadMe.md                                        # Project documentation
```

## 🚀 Features

- **Real-time Detection**: Live webcam feed processing
- **Video Processing**: Batch processing of video files
- **High Accuracy**: YOLOv8 nano model optimized for garbage detection
- **Fast Inference**: ~64ms per frame processing time
- **Visual Output**: Annotated results with bounding boxes and confidence scores

## 🛠️ Requirements

### Python Dependencies
```bash
pip install ultralytics
pip install opencv-python
pip install numpy
pip install matplotlib
pip install torch torchvision
pip install onnxruntime
```

### Hardware Requirements
- **Training**: Google Colab (Tesla T4 GPU recommended)
- **Inference**: Local machine with CPU/GPU support
- **Camera**: Webcam for real-time detection (optional)

## 📊 Dataset Structure

The project uses a custom dataset organized in YOLO format:

```
dataset/
├── train/
│   ├── images/          # Training images
│   └── labels/          # Training labels (.txt files)
├── test/
│   ├── images/          # Test images
│   └── labels/          # Test labels (.txt files)
└── valid/
    ├── images/          # Validation images
    └── labels/          # Validation labels (.txt files)
```

**Note**: Update `dataset.yaml` with your Google Drive paths for training in Colab.

## 🎯 Usage Instructions

### 1. Model Training (Google Colab)

1. **Open Colab Notebook**:
   - Upload `Garbage Detection Model Traning Process.ipynb` to Google Colab
   - Ensure GPU runtime is enabled (Runtime → Change runtime type → GPU)

2. **Prepare Dataset**:
   - Upload your dataset to Google Drive
   - Update paths in `dataset.yaml`:
   ```yaml
   train: /content/drive/MyDrive/your-path/train
   val: /content/drive/MyDrive/your-path/valid
   test: /content/drive/MyDrive/your-path/test
   ```

3. **Run Training**:
   - Execute all cells in the training notebook
   - Training will run for 35 epochs (~23 minutes)
   - Best model will be saved as `best.pt`

### 2. Local Inference & Testing

1. **Setup Environment**:
   ```bash
   # Install required packages
   pip install ultralytics opencv-python
   ```

2. **Real-time Webcam Detection**:
   - Open `real time detection and test video.ipynb`
   - Run the webcam detection cell
   - Press 'q' to quit the live feed

3. **Video Processing**:
   - Place your input video as `video.mp4`
   - Run the video processing cell
   - Output will be saved as `output_video.mp4`

## 📁 File Descriptions

### Training Files
- **`Garbage Detection Model Traning Process.ipynb`**: Complete training pipeline including data loading, model configuration, training execution, and validation
- **`dataset.yaml`**: Configuration file specifying dataset paths and class names for YOLO training

### Inference Files
- **`real time detection and test video.ipynb`**: Contains two main functions:
  - Real-time webcam detection
  - Video file processing with detection overlay
- **`best1.pt`**: Primary trained model (6.2MB, optimized for deployment)
- **`model1.pt`**: Alternative trained model checkpoint

### Media Files
- **`video.mp4`**: Sample input video for testing detection capabilities
- **`output_video_with_results.mp4`**: Example output showing detection results with bounding boxes and labels

## ⚙️ Model Configuration

### Training Parameters
- **Base Model**: YOLOv8 nano (yolov8n.pt)
- **Image Size**: 640x640 pixels
- **Batch Size**: 32
- **Epochs**: 35
- **Optimizer**: AdamW
- **Learning Rate**: 0.01 (initial)

### Performance Metrics
- **Inference Time**: ~64ms per frame
- **Model Size**: 6.2MB
- **Input Resolution**: 384x640 (optimized)
- **Training Time**: ~23 minutes on Tesla T4

## 🎥 Output Features

### Detection Visualization
- **Bounding Boxes**: Green rectangles around detected objects
- **Labels**: Class names with confidence scores
- **Real-time Display**: Live preview during processing
- **Video Export**: Save annotated videos for analysis

### Supported Formats
- **Input**: MP4, AVI, MOV, and other OpenCV-supported formats
- **Output**: MP4 with H.264 encoding
- **Camera**: USB webcams and built-in cameras

## 🔧 Customization

### Adding New Classes
1. Update your dataset with new class labels
2. Modify `dataset.yaml` with updated class names
3. Retrain the model using the training notebook

### Adjusting Detection Threshold
```python
# In inference code, modify confidence threshold
results = model.predict(source=frame, conf=0.5)  # Default: 0.25
```

### Changing Model Size
```python
# Use different YOLOv8 variants
model = YOLO('yolov8s.pt')  # Small
model = YOLO('yolov8m.pt')  # Medium
model = YOLO('yolov8l.pt')  # Large
```

## 🚨 Troubleshooting

### Common Issues

1. **Model Loading Error**:
   - Ensure `best1.pt` is in the same directory
   - Check file path in the code

2. **Camera Not Found**:
   - Change camera index: `cv2.VideoCapture(1)` or `cv2.VideoCapture(2)`
   - Check camera permissions

3. **CUDA/GPU Issues**:
   - Install appropriate PyTorch version for your CUDA version
   - Model will automatically fall back to CPU if GPU unavailable

4. **Memory Issues**:
   - Reduce batch size during training
   - Use smaller input image size

## 📈 Future Enhancements

- [ ] Add more garbage categories
- [ ] Implement waste sorting classification
- [ ] Deploy as web application
- [ ] Add mobile app support
- [ ] Integrate with IoT sensors
- [ ] Real-time analytics dashboard

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 📞 Contact

For questions, issues, or collaboration opportunities, please open an issue in the repository.

---

**Note**: This project is designed for educational and research purposes. For production deployment, consider additional optimizations and security