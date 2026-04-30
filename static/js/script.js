// Upload form handling
document.getElementById('uploadForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    const fileInput = document.getElementById('video');
    
    if (!fileInput.files[0]) {
        alert('Please select a video file');
        return;
    }
    
    // Show loading
    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('results').classList.add('hidden');
    document.getElementById('analyzeBtn').disabled = true;
    
    // Simulate progress (actual progress would come from server-sent events)
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += 5;
        if (progress <= 90) {
            document.getElementById('progress').style.width = progress + '%';
        }
    }, 500);
    
    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });
        
        clearInterval(progressInterval);
        document.getElementById('progress').style.width = '100%';
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data);
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        clearInterval(progressInterval);
        alert('Error uploading file: ' + error.message);
    } finally {
        document.getElementById('loading').classList.add('hidden');
        document.getElementById('analyzeBtn').disabled = false;
    }
});

// Display analysis results
function displayResults(data) {
    document.getElementById('framesCount').textContent = data.frames_processed;
    document.getElementById('gesturesCount').textContent = data.gestures_detected;
    
    document.getElementById('twoHandedCount').textContent = data.gesture_counts.TWO_HANDED_GESTURE || 0;
    document.getElementById('headMovementCount').textContent = data.gesture_counts.GESTURE_WITH_HEAD_MOVEMENT || 0;
    document.getElementById('handshapeCount').textContent = data.gesture_counts.HANDSHAPE_GESTURE || 0;
    
    // Display gestures
    const gesturesList = document.getElementById('gesturesList');
    gesturesList.innerHTML = '';
    
    if (data.gestures.length === 0) {
        gesturesList.innerHTML = '<p class="no-gestures">No gestures detected in the video.</p>';
    } else {
        data.gestures.forEach(gesture => {
            const gestureEl = createGestureElement(gesture);
            gesturesList.appendChild(gestureEl);
        });
    }
    
    // Display SiGML
    document.getElementById('sigmlContent').textContent = data.sigml;
    
    // Store for download
    window.currentSigML = data.sigml;
    
    // Show results
    document.getElementById('results').classList.remove('hidden');
}

// Create gesture element
function createGestureElement(gesture) {
    const div = document.createElement('div');
    div.className = 'gesture-item';
    
    const time = document.createElement('div');
    time.className = 'gesture-time';
    time.textContent = `Time: ${gesture.time}s (Frame: ${gesture.frame})`;
    
    const description = document.createElement('div');
    description.className = 'gesture-description';
    description.textContent = gesture.description;
    
    const details = document.createElement('div');
    details.className = 'gesture-details';
    
    // Add gesture type
    const typeSpan = document.createElement('span');
    typeSpan.className = 'gesture-type';
    typeSpan.textContent = gesture.type.replace(/_/g, ' ');
    details.appendChild(typeSpan);
    
    // Add hand shape if available
    if (gesture.hand_shape) {
        const shapeSpan = document.createElement('span');
        shapeSpan.textContent = `Shape: ${gesture.hand_shape}`;
        details.appendChild(shapeSpan);
    }
    
    // Add orientation if available
    if (gesture.hand_orientation) {
        const orientSpan = document.createElement('span');
        orientSpan.textContent = `Orientation: ${gesture.hand_orientation}`;
        details.appendChild(orientSpan);
    }
    
    // Add movement if available
    if (gesture.movement) {
        const moveSpan = document.createElement('span');
        moveSpan.textContent = `Movement: ${gesture.movement}`;
        details.appendChild(moveSpan);
    }
    
    // Add head movement if available
    if (gesture.head_movement) {
        const headSpan = document.createElement('span');
        headSpan.textContent = `Head: ${gesture.head_movement}`;
        details.appendChild(headSpan);
    }
    
    div.appendChild(time);
    div.appendChild(description);
    div.appendChild(details);
    
    return div;
}

// Tab switching
function showTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');
    
    // Update tab content
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.getElementById(tabName + 'Tab').classList.add('active');
}

// Download SiGML
function downloadSigML() {
    if (!window.currentSigML) return;
    
    const blob = new Blob([window.currentSigML], { type: 'application/xml' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'sign_language_analysis.sigml';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}

// Copy SiGML to clipboard
function copySigML() {
    if (!window.currentSigML) return;
    
    navigator.clipboard.writeText(window.currentSigML).then(() => {
        alert('SiGML copied to clipboard!');
    }).catch(err => {
        alert('Failed to copy: ' + err);
    });
}

// File input validation
document.getElementById('video').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        // Check file size (100MB limit)
        if (file.size > 100 * 1024 * 1024) {
            alert('File size exceeds 100MB limit');
            this.value = '';
            return;
        }
        
        // Check file type
        const validTypes = ['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/x-matroska'];
        if (!validTypes.includes(file.type) && !file.name.match(/\.(mp4|mov|avi|mkv)$/i)) {
            alert('Please upload a valid video file (MP4, MOV, AVI, MKV)');
            this.value = '';
            return;
        }
    }
});
