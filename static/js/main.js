const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const startCameraBtn = document.getElementById('start-camera');
const captureBtn = document.getElementById('capture');
const resultSection = document.getElementById('result-section');
const capturedImage = document.getElementById('captured-image');
const predictionResult = document.getElementById('prediction-result');
const loading = document.getElementById('loading');

let stream = null;

// Start camera
startCameraBtn.addEventListener('click', async () => {
    try {
        // Request camera access with user-facing camera
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                facingMode: 'user',  // Front camera
                width: { ideal: 640 },
                height: { ideal: 480 }
            } 
        });
        
        video.srcObject = stream;
        captureBtn.disabled = false;
        startCameraBtn.textContent = '✓ Camera Active';
        startCameraBtn.disabled = true;
        startCameraBtn.style.background = '#28a745';
        
        console.log('Camera started successfully');
        
    } catch (error) {
        console.error('Error accessing camera:', error);
        alert('Error accessing camera: ' + error.message + '\n\nPlease make sure you have granted camera permissions.');
    }
});

// Capture and predict
captureBtn.addEventListener('click', async () => {
    console.log('Capturing image...');
    
    // Set canvas dimensions to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw video frame to canvas
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert canvas to base64 image
    const imageData = canvas.toDataURL('image/jpeg');
    
    // Display captured image
    capturedImage.src = imageData;
    resultSection.style.display = 'block';
    loading.style.display = 'block';
    predictionResult.textContent = '';
    predictionResult.className = '';
    
    try {
        console.log('Sending image to server...');
        
        // Send to backend for prediction
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData })
        });
        
        const result = await response.json();
        
        loading.style.display = 'none';
        
        if (result.error) {
            console.error('Prediction error:', result.error);
            predictionResult.textContent = 'Error: ' + result.error;
            predictionResult.className = '';
        } else {
            console.log('Prediction result:', result);
            
            const maskStatus = result.wearing_mask ? 'Wearing Mask ✓' : 'No Mask ✗';
            predictionResult.textContent = `${maskStatus} (${result.confidence}% confident)`;
            predictionResult.className = result.wearing_mask ? 'with-mask' : 'without-mask';
        }
        
    } catch (error) {
        console.error('Error during prediction:', error);
        loading.style.display = 'none';
        predictionResult.textContent = 'Error: ' + error.message;
        predictionResult.className = '';
    }
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
});
