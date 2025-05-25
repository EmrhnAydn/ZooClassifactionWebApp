// Main JavaScript file for Multi Zoo Classification

document.addEventListener('DOMContentLoaded', () => {
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const previewImage = document.getElementById('preview-image');
    const uploadPrompt = document.getElementById('upload-prompt');
    const resultContainer = document.getElementById('result-container');
    const resultsBody = document.getElementById('results-body');
    const loading = document.getElementById('loading');
    const selectBtn = document.getElementById('select-btn');
    const refreshBtn = document.getElementById('refresh-btn');
    const testDataInput = document.getElementById('test-data-input');
    const testDataBtn = document.getElementById('test-data-btn');
    const testDataStatus = document.getElementById('test-data-status');
    const toggleTestBtn = document.getElementById('toggle-test-btn');
    const testDataSection = document.getElementById('test-data-section');
    const testResults = document.getElementById('test-results');
    
    // Constants for batch processing
    const BATCH_SIZE = 5; // Number of images to upload in each batch
    let currentTestId = null;
    let statusCheckInterval = null;

    // Handle file select button
    selectBtn.addEventListener('click', () => {
        fileInput.click();
    });

    // Handle refresh button - reset the UI to upload a new image
    refreshBtn.addEventListener('click', () => {
        resetUI();
    });

    // Toggle test data section
    toggleTestBtn.addEventListener('click', () => {
        testDataSection.classList.toggle('d-none');
        if (testDataSection.classList.contains('d-none')) {
            toggleTestBtn.textContent = 'Modeli Test Et';
        } else {
            toggleTestBtn.textContent = 'Test Bölümünü Gizle';
        }
    });

    // Function to reset UI to initial state for new upload
    function resetUI() {
        fileInput.value = '';
        previewImage.src = '';
        previewContainer.classList.add('d-none');
        uploadPrompt.classList.remove('d-none');
        resultContainer.classList.add('d-none');
    }

    // Handle test data upload
    testDataBtn.addEventListener('click', () => {
        if (testDataInput.files.length > 0) {
            // Show loading state
            testDataBtn.disabled = true;
            testDataBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Yükleniyor...';
            testDataStatus.classList.remove('d-none');
            testDataStatus.querySelector('.alert').innerHTML = 'Test verisi hazırlanıyor...';
            testResults.classList.add('d-none');
            
            // Clear any existing status check
            if (statusCheckInterval) {
                clearInterval(statusCheckInterval);
                statusCheckInterval = null;
            }
            
            // Process files
            const files = Array.from(testDataInput.files);
            
            // Organize files by class
            const filesByClass = organizeFilesByClass(files);
            
            // Start the test with batch processing
            startBatchProcessing(filesByClass);
        } else {
            alert('Lütfen test veri klasörünü seçin.');
        }
    });
    
    // Organize uploaded files into class structure
    function organizeFilesByClass(files) {
        const filesByClass = {};
        
        files.forEach(file => {
            const path = file.webkitRelativePath;
            const pathParts = path.split('/');
            
            // Expecting structure: Test/className/image.jpg
            if (pathParts.length >= 3) {
                const className = pathParts[1];
                
                // Only process image files
                if (file.type.startsWith('image/')) {
                    if (!filesByClass[className]) {
                        filesByClass[className] = [];
                    }
                    filesByClass[className].push(file);
                }
            }
        });
        
        return filesByClass;
    }
    
    // Start batch processing of test data
    async function startBatchProcessing(filesByClass) {
        // Initialize the test on the server
        const classes = Object.keys(filesByClass);
        if (classes.length === 0) {
            handleTestError('Geçerli test görüntüsü bulunamadı. Lütfen doğru klasör yapısını kullandığınızdan emin olun.');
            return;
        }
        
        // Create form data with class information and file counts
        const formData = new FormData();
        formData.append('classes', JSON.stringify(classes));
        
        // Add file counts for each class
        let totalFiles = 0;
        classes.forEach(className => {
            const count = filesByClass[className].length;
            formData.append(`${className}_count`, count);
            totalFiles += count;
        });
        
        if (totalFiles === 0) {
            handleTestError('Geçerli test görüntüsü bulunamadı.');
            return;
        }
        
        try {
            // Start the test
            const response = await fetch('/start_test', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('Test başlatılamadı');
            }
            
            const data = await response.json();
            currentTestId = data.test_id;
            
            // Update status message
            testDataStatus.querySelector('.alert').innerHTML = 
                `<div>Test işlemi başlatıldı. Toplam dosya: ${totalFiles}</div>
                 <div class="progress mt-2" style="height: 20px;">
                     <div class="progress-bar progress-bar-striped progress-bar-animated" 
                          role="progressbar" style="width: 0%" 
                          aria-valuenow="0" aria-valuemin="0" aria-valuemax="100">0%</div>
                 </div>`;
            
            // Start uploading batches
            for (const className of classes) {
                await uploadClassBatches(className, filesByClass[className]);
            }
            
            // Start checking status
            startStatusChecking();
            
        } catch (error) {
            handleTestError(`Test başlatılırken hata oluştu: ${error.message}`);
        }
    }
    
    // Upload files for a class in batches
    async function uploadClassBatches(className, files) {
        // Split files into batches
        for (let i = 0; i < files.length; i += BATCH_SIZE) {
            const batch = files.slice(i, i + BATCH_SIZE);
            await uploadBatch(className, batch);
        }
    }
    
    // Upload a single batch of files
    async function uploadBatch(className, batchFiles) {
        const formData = new FormData();
        formData.append('test_id', currentTestId);
        formData.append('class_name', className);
        
        // Add files to form data
        batchFiles.forEach((file, index) => {
            formData.append(`file_${index}`, file);
        });
        
        try {
            const response = await fetch('/upload_test_batch', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('Batch upload failed');
            }
            
            const data = await response.json();
            updateProgressBar(data.progress, data.total);
            
        } catch (error) {
            console.error('Error uploading batch:', error);
            // Continue with other batches even if one fails
        }
    }
    
    // Start checking test status periodically
    function startStatusChecking() {
        if (statusCheckInterval) {
            clearInterval(statusCheckInterval);
        }
        
        statusCheckInterval = setInterval(async () => {
            try {
                const response = await fetch(`/test_status/${currentTestId}`);
                
                if (!response.ok) {
                    throw new Error('Status check failed');
                }
                
                const data = await response.json();
                updateProgressBar(data.progress, data.total);
                
                // Check if completed
                if (data.status === 'completed') {
                    clearInterval(statusCheckInterval);
                    statusCheckInterval = null;
                    testDataBtn.disabled = false;
                    testDataBtn.innerHTML = 'Yükle';
                    displayTestResults(data.results);
                }
                
                // Check if error
                if (data.status === 'error') {
                    clearInterval(statusCheckInterval);
                    statusCheckInterval = null;
                    handleTestError(`Test işleminde hata: ${data.error || 'Bilinmeyen hata'}`);
                }
                
            } catch (error) {
                console.error('Error checking status:', error);
                // Don't stop checking on a single error
            }
        }, 2000); // Check every 2 seconds
    }
    
    // Update progress bar
    function updateProgressBar(progress, total) {
        const percentage = Math.min(100, Math.round((progress / total) * 100));
        const progressBar = testDataStatus.querySelector('.progress-bar');
        if (progressBar) {
            progressBar.style.width = `${percentage}%`;
            progressBar.setAttribute('aria-valuenow', percentage);
            progressBar.textContent = `${percentage}%`;
        }
    }
    
    // Handle test error
    function handleTestError(message) {
        testDataBtn.disabled = false;
        testDataBtn.innerHTML = 'Yükle';
        testDataStatus.querySelector('.alert').innerHTML = 
            `<div class="text-danger">${message}</div>`;
    }
    
    // Display test results
    function displayTestResults(data) {
        // Show the results section
        testResults.classList.remove('d-none');
        
        // Update metrics
        document.getElementById('accuracy-value').textContent = `${(data.accuracy * 100).toFixed(2)}%`;
        document.getElementById('precision-value').textContent = `${(data.precision * 100).toFixed(2)}%`;
        document.getElementById('recall-value').textContent = `${(data.recall * 100).toFixed(2)}%`;
        document.getElementById('f1-value').textContent = `${(data.f1 * 100).toFixed(2)}%`;
        
        // Update status message
        testDataStatus.querySelector('.alert').innerHTML = 
            `<div class="text-success">Test tamamlandı! Toplam ${data.total_images} görüntü işlendi.</div>`;
    }

    // Handle file selection
    fileInput.addEventListener('change', handleFile);

    // Drag and drop events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });

    dropArea.addEventListener('drop', handleDrop, false);

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function highlight() {
        dropArea.classList.add('bg-dark');
    }

    function unhighlight() {
        dropArea.classList.remove('bg-dark');
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;

        if (files.length) {
            fileInput.files = files;
            handleFile();
        }
    }

    function handleFile() {
        const file = fileInput.files[0];
        if (!file) return;

        // Display preview
        const reader = new FileReader();
        reader.onload = function(e) {
            previewImage.src = e.target.result;
            previewContainer.classList.remove('d-none');
            uploadPrompt.classList.add('d-none');
        };
        reader.readAsDataURL(file);

        // Upload and classify
        uploadAndClassify(file);
    }

    function uploadAndClassify(file) {
        resultContainer.classList.add('d-none');
        loading.classList.remove('d-none');

        const formData = new FormData();
        formData.append('file', file);

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            resultsBody.innerHTML = '';
            data.forEach(result => {
                const percent = (result.prob * 100).toFixed(2);
                resultsBody.innerHTML += `
                    <tr>
                        <td>${result.label}</td>
                        <td>
                            <div class="d-flex align-items-center">
                                <div class="progress flex-grow-1 me-2" style="height: 10px;">
                                    <div class="progress-bar" style="width: ${percent}%"></div>
                                </div>
                                <span>${percent}%</span>
                            </div>
                        </td>
                    </tr>
                `;
            });
            resultContainer.classList.remove('d-none');
            loading.classList.add('d-none');
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Görsel yüklenirken bir hata oluştu.');
            loading.classList.add('d-none');
        });
    }
}); 