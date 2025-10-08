from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import base64
import os
from io import BytesIO
from PIL import Image
from keras.models import load_model
from keras import Model, layers
from pyngrok import ngrok

app = Flask(__name__)
CORS(app)

# =========================================================
# Configuration - MODIFIED TO ADD INVERSION
# =========================================================
IMG_SIZE = 128
CATEGORIES = ["WithMask", "WithoutMask"]
CONF_THRESHOLD = 0.85                        # CHANGED: Lowered from 0.90
SKIN_RATIO_FORCE_NO_MASK = 0.35
INVERT_PREDICTIONS = True                    # ← NEW: Enable inversion

# Define the custom ConvNet class
class ConvNet(Model):
    def __init__(self, **kwargs):
        super(ConvNet, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')
        self.pool1 = layers.MaxPooling2D()
        self.conv2 = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')
        self.pool2 = layers.MaxPooling2D()
        self.conv3 = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')
        self.pool3 = layers.MaxPooling2D()
        self.conv4 = layers.Conv2D(128, kernel_size=(3, 3), activation='relu')
        self.pool4 = layers.MaxPooling2D()
        self.conv5 = layers.Conv2D(128, kernel_size=(3, 3), activation='relu')
        self.pool5 = layers.MaxPooling2D()
        self.flatten = layers.Flatten()
        self.dropout = layers.Dropout(0.3)
        self.classifier = layers.Dense(2, activation='softmax')

    def call(self, x):
        x = self.conv1(x); x = self.pool1(x)
        x = self.conv2(x); x = self.pool2(x)
        x = self.conv3(x); x = self.pool3(x)
        x = self.conv4(x); x = self.pool4(x)
        x = self.conv5(x); x = self.pool5(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return self.classifier(x)

    def get_config(self):
        return super(ConvNet, self).get_config()

# Model path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'convnet.keras')
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Debug information - MODIFIED
print(f"\n{'='*60}")
print("STARTUP CHECKS")
print(f"{'='*60}")
print(f"Model file exists: {os.path.exists(MODEL_PATH)}")
print(f"Face cascade loaded: {not FACE_CASCADE.empty()}")
print(f"Prediction inversion: {'ENABLED ✓' if INVERT_PREDICTIONS else 'DISABLED'}")  # ← NEW
print(f"{'='*60}\n")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

model = load_model(MODEL_PATH, custom_objects={'ConvNet': ConvNet})
print("✓ Model loaded successfully")
if INVERT_PREDICTIONS:  # ← NEW
    print("⚠ WARNING: Predictions will be inverted (model labels are backwards)\n")

# Helper functions (unchanged)
def _largest_face_bbox(gray_img):
    faces = FACE_CASCADE.detectMultiScale(
        gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2]*f[3])

def _apply_padding(x, y, w, h, H, W, pad_ratio=0.30):
    ph = int(h * pad_ratio); pw = int(w * pad_ratio)
    y1 = max(0, y - ph); y2 = min(H, y + h + ph)
    x1 = max(0, x - pw); x2 = min(W, x + w + pw)
    return x1, y1, x2, y2

def _lower_face_roi(bgr_face):
    h, w = bgr_face.shape[:2]
    y0 = int(h * 0.45)
    return bgr_face[y0:h, :]

def _skin_ratio(bgr_img):
    if bgr_img.size == 0:  # ← NEW: Safety check
        return 0.0
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 30, 50], dtype=np.uint8)
    upper1 = np.array([20, 180, 255], dtype=np.uint8)
    lower2 = np.array([160, 30, 50], dtype=np.uint8)
    upper2 = np.array([179, 180, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    return float(np.count_nonzero(mask)) / float(mask.size)

def preprocess_image(image):
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    bbox = _largest_face_bbox(gray)
    if bbox is not None:
        print(f"✓ Detected face(s)")
        x, y, w, h = bbox
        H, W = gray.shape[:2]
        x1, y1, x2, y2 = _apply_padding(x, y, w, h, H, W, pad_ratio=0.30)
        gray = gray[y1:y2, x1:x2]
        print(f"✓ Face region cropped: ({x1}, {y1}) to ({x2}, {y2})")
    else:
        print("⚠ No face detected - using full image")

    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    normalized = resized.astype(np.float32) / 255.0
    return normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    MODIFIED: Added prediction inversion and low-skin detection
    """
    try:
        data = request.get_json()
        image_data = data['image']
        image_data = image_data.split(',')[1]
        image = Image.open(BytesIO(base64.b64decode(image_data))).convert("RGB")

        print("\n" + "="*60)
        print("PROCESSING NEW IMAGE")
        print("="*60)

        # ---- Skin detection ----
        rgb_np = np.array(image)
        bgr_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
        gray_full = cv2.cvtColor(bgr_np, cv2.COLOR_BGR2GRAY)
        bbox = _largest_face_bbox(gray_full)
        skin_pct = 0.0
        
        if bbox is not None:
            x, y, w, h = bbox
            H, W = gray_full.shape[:2]
            x1, y1, x2, y2 = _apply_padding(x, y, w, h, H, W, pad_ratio=0.30)
            face_bgr = bgr_np[y1:y2, x1:x2]
            roi_bgr = _lower_face_roi(face_bgr)
            if roi_bgr.size > 0:
                skin_pct = _skin_ratio(roi_bgr)
        
        print(f"🔍 Skin ratio in lower-face: {skin_pct:.3f}")

        # ---- Model prediction ----
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image, verbose=0)[0]
        original_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        print(f"\n📊 Raw model output:")
        print(f"   Class {original_class}: {CATEGORIES[original_class]}")
        print(f"   Probabilities: WithMask={prediction[0]*100:.2f}%, WithoutMask={prediction[1]*100:.2f}%")

        # ========== INVERSION LOGIC - THIS IS THE KEY FIX ==========
        if INVERT_PREDICTIONS:
            predicted_class = 1 - original_class  # Flip 0→1, 1→0
            with_mask_prob = float(prediction[1] * 100)  # Swapped
            without_mask_prob = float(prediction[0] * 100)  # Swapped
            print(f"\n🔄 INVERTING PREDICTION:")
            print(f"   Model said: {CATEGORIES[original_class]}")
            print(f"   Corrected to: {CATEGORIES[predicted_class]}")
        else:
            predicted_class = original_class
            with_mask_prob = float(prediction[0] * 100)
            without_mask_prob = float(prediction[1] * 100)
        # ============================================================

        # ---- Skin detection override ----
        forced_by_skin = False
        
        # NEW: Check for LOW skin (mask present)
        if skin_pct < 0.15:
            print(f"✓ LOW SKIN ({skin_pct:.3f}) → Confirmed WithMask")
            predicted_class = CATEGORIES.index("WithMask")
            forced_by_skin = True
        # Check for HIGH skin (no mask)
        elif skin_pct >= SKIN_RATIO_FORCE_NO_MASK:
            print(f"✓ HIGH SKIN ({skin_pct:.3f}) → Forcing WithoutMask")
            predicted_class = CATEGORIES.index("WithoutMask")
            forced_by_skin = True

        final_label = CATEGORIES[predicted_class]
        final_conf = confidence
        wearing_mask = (final_label == "WithMask")

        # Confidence gate
        if not forced_by_skin and confidence < CONF_THRESHOLD:
            print(f"⚠ Low confidence ({confidence*100:.1f}%) → Uncertain")
            final_label = "Uncertain"
            wearing_mask = False

        # Prepare result
        result = {
            'prediction': final_label,
            'confidence': round(final_conf * 100, 2),
            'wearing_mask': bool(wearing_mask),
            'probabilities': {
                'WithMask': round(with_mask_prob, 2),
                'WithoutMask': round(without_mask_prob, 2)
            }
        }

        print(f"\n✅ FINAL: {result['prediction']} ({result['confidence']}%)")
        print(f"   Wearing mask: {result['wearing_mask']}")
        print(f"   Method: {'Skin detection' if forced_by_skin else 'Model prediction'}")
        print("="*60 + "\n")

        return jsonify(result)

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'face_cascade_loaded': not FACE_CASCADE.empty(),
        'inversion_enabled': INVERT_PREDICTIONS
    })

# if __name__ == '__main__':
#     print("\n🚀 Starting Flask server...")
#     print("📷 Face Mask Detection App")
#     print("🌐 Local: http://localhost:5000")
#     print("=" * 60)
#     app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == '__main__':
    ngrok.set_auth_token("33mBDyjHzpP7DSdfR4aLN0utyYS_7Fcm27XpjXkzSYzzwSAs3")  # Replace with your token
    
    # Start ngrok tunnel
    public_url = ngrok.connect(5000)
    
    print("FACE MASK DETECTION SERVER")
    print("\n📋 Access URLs:")
    print(f"   🏠 Local:  http://localhost:5000")
    print(f"   🌐 Public: {public_url}")
    print("\n⚠️  Camera will work on both URLs!")
    print("   Share the public URL with anyone!\n")
    print("=" * 60 + "\n")
    
    # app.run(debug=True, host='0.0.0.0', port=5000)
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)




