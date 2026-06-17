document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const fileInput = document.getElementById('file-input');
    const captureBtn = document.getElementById('capture-btn');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const analyzeBtn = document.getElementById('analyze-btn');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const newAnalysisBtn = document.getElementById('new-analysis-btn');
    
    // Result elements
    const resultImage = document.getElementById('result-image');
    const diseaseName = document.getElementById('disease-name');
    const diseaseBadge = document.getElementById('disease-badge');
    const confidenceValue = document.getElementById('confidence-value');
    const confidenceBar = document.getElementById('confidence-bar');
    const causesText = document.getElementById('causes-text');
    const symptomsText = document.getElementById('symptoms-text');
    const preventionList = document.getElementById('prevention-list');
    const treatmentList = document.getElementById('treatment-list');
    const severity = document.getElementById('severity');
    
    // Tab elements
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    // Current file for analysis
    let currentFile = null;
    
    // Event Listeners
    fileInput.addEventListener('change', handleFileSelect);
    captureBtn.addEventListener('click', handleCapture);
    analyzeBtn.addEventListener('click', analyzeImage);
    newAnalysisBtn.addEventListener('click', resetAnalysis);
    
    // Tab switching
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.tab;
            switchTab(tab);
        });
    });
    
    function handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            currentFile = file;
            displayPreview(file);
        }
    }
    
    function handleCapture() {
        if ('mediaDevices' in navigator && 'getUserMedia' in navigator.mediaDevices) {
            fileInput.setAttribute('capture', 'environment');
            fileInput.click();
        } else {
            alert('Camera not supported on this device');
            fileInput.removeAttribute('capture');
            fileInput.click();
        }
    }
    
    function displayPreview(file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            imagePreview.src = e.target.result;
            previewContainer.classList.remove('hidden');
        }
        reader.readAsDataURL(file);
    }
    
    async function analyzeImage() {
        if (!currentFile) {
            alert('Please select an image first');
            return;
        }
        
        // Show loading, hide results and preview
        loading.classList.remove('hidden');
        results.classList.add('hidden');
        previewContainer.classList.add('hidden');
        
        // Create form data
        const formData = new FormData();
        formData.append('file', currentFile);
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.success) {
                displayResults(data);
            } else {
                alert('Error: ' + data.error);
                resetAnalysis();
            }
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while analyzing the image');
            resetAnalysis();
        } finally {
            loading.classList.add('hidden');
        }
    }
    
    function displayResults(data) {
        // Set image
        resultImage.src = data.image_url;
        
        // Set disease name and badge
        const diseaseKey = data.disease.toLowerCase().replace(' ', '_');
        diseaseName.textContent = data.info.name;
        diseaseBadge.className = `disease-badge ${diseaseKey}`;
        
        // Set confidence
        const confidencePercent = Math.round(data.confidence * 100);
        confidenceValue.textContent = `${confidencePercent}%`;
        confidenceBar.style.width = `${confidencePercent}%`;
        
        // Set disease information
        causesText.textContent = data.info.causes;
        symptomsText.textContent = data.info.symptoms;
        
        // Set prevention list
        preventionList.innerHTML = '';
        data.info.prevention.forEach(item => {
            const li = document.createElement('li');
            li.innerHTML = `<i class="fas fa-check-circle" style="color: var(--success-color); margin-right: 8px;"></i>${item}`;
            preventionList.appendChild(li);
        });
        
        // Set treatment list
        treatmentList.innerHTML = '';
        data.info.treatment.forEach(item => {
            const li = document.createElement('li');
            li.innerHTML = `<i class="fas fa-flask" style="color: var(--primary-color); margin-right: 8px;"></i>${item}`;
            treatmentList.appendChild(li);
        });
        
        // Set severity
        severity.textContent = data.info.severity;
        
        // Show results
        results.classList.remove('hidden');
        
        // Reset to first tab
        switchTab('causes');
    }
    
    function switchTab(tabId) {
        // Update tab buttons
        tabBtns.forEach(btn => {
            if (btn.dataset.tab === tabId) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });
        
        // Update tab content
        tabContents.forEach(content => {
            if (content.id === `${tabId}-tab`) {
                content.classList.add('active');
            } else {
                content.classList.remove('active');
            }
        });
    }
    
    function resetAnalysis() {
        // Reset file input
        fileInput.value = '';
        currentFile = null;
        
        // Hide preview, loading, results
        previewContainer.classList.add('hidden');
        loading.classList.add('hidden');
        results.classList.add('hidden');
        
        // Show upload section
        document.querySelector('.upload-section').scrollIntoView({ behavior: 'smooth' });
    }
});