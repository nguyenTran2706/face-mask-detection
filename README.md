# Face Mask Detection Web Application

A real-time face mask detection system using Flask, TensorFlow/Keras, and OpenCV. The application uses a Convolutional Neural Network (CNN) to classify whether a person is wearing a mask or not through webcam feed.

## Features

- 🎥 Real-time face mask detection via webcam
- 🧠 CNN-based deep learning model
- 👤 Face detection using Haar Cascade Classifier
- 🔄 Automatic prediction inversion for correcting model labels
- 🌐 Public URL support via ngrok for remote access
- 🎨 Clean, responsive web interface
- 📊 Live confidence scores and probability display

## Technologies Used

- **Backend**: Flask, Python 3.13
- **Machine Learning**: TensorFlow, Keras, NumPy
- **Computer Vision**: OpenCV
- **Frontend**: HTML5, JavaScript, CSS
- **Networking**: Flask-CORS, pyngrok

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8+** (Recommended: Python 3.13)
- **pip** (Python package manager)
- **Git** (for cloning the repository)
- **Webcam** (for real-time detection)

### Check Python Installation
python --version
Should return `Python 3.x.x`

## Installation & Setup
### Step 1: Clone the Repository
git clone https://github.com/nguyenTran2706/face-mask-detection.git
cd face-mask-detection


### Step 2: Create Virtual Environment
**Windows:** 
python -m venv .venv
..venv\Scripts\Activate.ps1

**Mac/Linux:**
python3 -m venv .venv
source .venv/bin/activate
You should see `(.venv)` at the start of your terminal prompt.

### Step 3: Install Required Libraries
Install all dependencies from requirements.txt:
pip install -r requirements.txt
**Or install manually:**
pip install flask flask-cors numpy opencv-python pillow tensorflow keras pyngrok


**Dependency Breakdown:**
- `flask` - Web framework
- `flask-cors` - Cross-Origin Resource Sharing support
- `numpy` - Numerical computations
- `opencv-python` - Computer vision and image processing
- `pillow` - Image handling
- `tensorflow` - Deep learning framework
- `keras` - High-level neural networks API
- `pyngrok` - Public URL tunneling (optional)

### Step 4: Configure ngrok (Optional - For Public Access)
If you want to access the app from anywhere (not just localhost):
1. **Sign up** for a free ngrok account: https://ngrok.com/signup
2. **Get your authtoken**: https://dashboard.ngrok.com/get-started/your-authtoken
3. **Edit `app.py`** and replace the placeholder:
   
ngrok.set_auth_token("YOUR_NGROK_TOKEN") # Replace with your actual token

**Note:** If you only want local access, you can comment out or remove the ngrok-related code.

### Step 5: Verify Model File

Ensure `convnet.keras` is in the project root directory. This is the pre-trained CNN model required for predictions.

## Running the Application

### Start the Server
python app.py

You should see output like:
============================================================
STARTUP CHECKS
Model file exists: True
Face cascade loaded: True
Prediction inversion: ENABLED ✓
✓ Model loaded successfully
⚠ WARNING: Predictions will be inverted (model labels are backwards)

🚀🚀🚀 FACE MASK DETECTION SERVER 🚀🚀🚀

📋 Access URLs:
🏠 Local: http://localhost:5000
🌐 Public: https://xyz.ngrok-free.dev (if ngrok enabled)

⚠️ Camera will work on both URLs!
Running on http://127.0.0.1:5000

Running on http://192.168.0.191:5000

### Access the Application

#### Local Access (Your Computer)
Open your browser and go to:
http://localhost:5000
#### Network Access (Same WiFi)
From other devices on the same network:
http://YOUR_IP_ADDRESS:5000
Replace `YOUR_IP_ADDRESS` with your computer's IP (shown in the terminal output)

#### Public Access (Anywhere via ngrok)
If ngrok is configured:
https://xyz.ngrok-free.dev
Share this URL with anyone!

### Using the Application

1. **Allow Camera Access**: Browser will prompt for camera permission - click "Allow"
2. **Start Detection**: Click "Start Camera" button
3. **View Results**: See real-time predictions with confidence scores
4. **Capture**: Click "Capture" to freeze and analyze a frame

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'flask'"

**Solution:** Make sure virtual environment is activated and dependencies are installed:
..venv\Scripts\Activate.ps1 # Windows
pip install -r requirements.txt

### Issue: "No module named 'pyngrok'"

**Solution:** Install pyngrok:
pip install pyngrok


Or remove ngrok code if you don't need public access.

### Issue: Camera not working on network devices

**Solution:** Camera requires HTTPS on non-localhost connections. Use ngrok or access via `localhost` only.

### Issue: "ModuleNotFoundError: No module named 'numpy'"

**Solution:** Install all dependencies:
pip install numpy tensorflow keras opencv-python


### Issue: Model predictions are inverted

**Solution:** The code includes automatic prediction inversion. Check that `INVERT_PREDICTIONS = True` in `app.py`.

### Issue: Port 5000 already in use

**Solution:** Change the port in `app.py`:
app.run(debug=True, host='0.0.0.0', port=5001) # Use 5001 instead


## Configuration Options

In `app.py`, you can adjust these parameters:

IMG_SIZE = 128 # Input image size

CONF_THRESHOLD = 0.85 # Minimum confidence threshold

SKIN_RATIO_FORCE_NO_MASK = 0.35 # Skin detection threshold

INVERT_PREDICTIONS = True # Enable prediction inversion

## Model Information

- **Architecture**: Custom Convolutional Neural Network (ConvNet)
- **Input Size**: 128x128 grayscale images
- **Output**: Binary classification (WithMask / WithoutMask)
- **Training**: Trained on face mask dataset with data augmentation
- **Features**: 
  - 5 Convolutional layers
  - MaxPooling layers
  - Dropout regularization
  - Softmax activation

## Security Notes

⚠️ **Important Security Considerations:**

1. **Never commit your ngrok authtoken** to public repositories
2. **Development server**: This uses Flask's development server - not suitable for production
3. **For production deployment**, use a WSGI server like Gunicorn or uWSGI
4. **CORS is enabled** for all origins - restrict this in production

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is open source and available under the MIT License.

## Contact

**Project Author**: Nguyen Tran  
**GitHub**: https://github.com/nguyenTran2706  
**Repository**: https://github.com/nguyenTran2706/face-mask-detection

## Acknowledgments

- TensorFlow and Keras for deep learning framework
- OpenCV for computer vision capabilities
- Flask for web framework
- ngrok for public URL tunneling








