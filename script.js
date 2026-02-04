// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// DOM Elements
let currentTab = 'youtube';

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    // Set up file input change handler
    document.getElementById('pdf-file').addEventListener('change', function(e) {
        const fileName = e.target.files[0]?.name || '';
        document.getElementById('file-name').textContent = fileName;
    });

    // Toggle bullet count control
    const summaryTypeEl = document.getElementById('summary-type');
    const bulletGroupEl = document.getElementById('bullet-count-group');
    const toggleBulletGroup = () => {
        if (summaryTypeEl.value === 'bullet') {
            bulletGroupEl.style.display = '';
        } else {
            bulletGroupEl.style.display = 'none';
        }
    };
    summaryTypeEl.addEventListener('change', toggleBulletGroup);
    toggleBulletGroup();
});

// Tab switching functionality
function switchTab(tabName) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active class from all tab buttons
    document.querySelectorAll('.tab-button').forEach(button => {
        button.classList.remove('active');
    });
    
    // Show selected tab content
    document.getElementById(tabName + '-tab').classList.add('active');
    
    // Add active class to clicked button
    event.target.classList.add('active');
    
    currentTab = tabName;
    
    // Clear previous results
    hideAllSections();
}

// Unified summarize handler placed after options section
function handleSummarize() {
    if (currentTab === 'youtube') {
        summarizeYouTube();
    } else if (currentTab === 'pdf') {
        summarizePDF();
    }
}

// Hide all result sections
function hideAllSections() {
    document.getElementById('loading').classList.add('hidden');
    document.getElementById('results').classList.add('hidden');
    document.getElementById('error').classList.add('hidden');
}

// Show loading state
function showLoading() {
    hideAllSections();
    document.getElementById('loading').classList.remove('hidden');
}

// Show error message
function showError(message) {
    hideAllSections();
    document.getElementById('error-message').textContent = message;
    document.getElementById('error').classList.remove('hidden');
}

// Show results
function showResults(data) {
    hideAllSections();
    
    const summaryText = document.getElementById('summary-text');
    const summaryInfo = document.getElementById('summary-info');
    
    // summary may be HTML
    summaryText.innerHTML = data.summary;
    
    // Create info text based on data type
    let infoText = '';
    if (data.video_url) {
        infoText = `Video URL: ${data.video_url}\nTranscript Length: ${data.transcript_length} characters`;
    } else if (data.filename) {
        infoText = `File: ${data.filename}\nText Length: ${data.text_length} characters`;
    }
    
    summaryInfo.textContent = infoText;
    document.getElementById('results').classList.remove('hidden');
}

// Copy summary to clipboard
function copySummary() {
    const source = document.getElementById('summary-text');
    if (!source) return showError("No summary to copy");

    // Create a clean version with forced bullet dots
    const clone = source.cloneNode(true);
    clone.querySelectorAll("li").forEach(li => {
        if (!li.innerText.trim().startsWith("•")) {
            li.innerText = "• " + li.innerText.trim();
        }
    });

    const html = clone.innerHTML;
    const plain = clone.innerText;

    // Modern Clipboard API
    if (navigator.clipboard && window.ClipboardItem) {
        try {
            const item = new ClipboardItem({
                "text/html": new Blob([html], { type: "text/html" }),
                "text/plain": new Blob([plain], { type: "text/plain" })
            });
            navigator.clipboard.write([item]).then(() => flashCopySuccess());
            return;
        } catch (err) {
            console.warn("ClipboardItem failed:", err);
        }
    }

    // Fallback copy
    const temp = document.createElement("div");
    temp.contentEditable = true;
    temp.style.position = "fixed";
    temp.style.left = "-9999px";
    temp.innerHTML = html;
    document.body.appendChild(temp);

    const sel = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(temp);
    sel.removeAllRanges();
    sel.addRange(range);

    document.execCommand("copy");
    sel.removeAllRanges();
    temp.remove();

    flashCopySuccess();
}

function flashCopySuccess() {
    const btn = document.querySelector(".copy-btn");
    const old = btn.innerHTML;
    btn.innerHTML = "<i class='fas fa-check'></i> Copied!";
    btn.style.background = "#1f8a3f";
    setTimeout(() => {
        btn.innerHTML = old;
        btn.style.background = "#28a745";
    }, 1500);
}


// Validate YouTube URL
function isValidYouTubeUrl(url) {
    const pattern = /^(https?:\/\/)?(www\.)?(youtube\.com\/(watch\?v=|shorts\/)\w[-\w]{10,}|youtu\.be\/\w[-\w]{10,})/i;
    return pattern.test(url);
}

// Summarize YouTube video
async function summarizeYouTube() {
    const url = document.getElementById('youtube-url').value.trim();
    const summaryType = document.getElementById('summary-type').value;
    const aiProvider = 'gemini';
    const bulletCountEl = document.getElementById('bullet-count');
    const bulletCount = bulletCountEl ? bulletCountEl.value : '';
    
    // Validation
    if (!url) {
        showError('Please enter a YouTube URL');
        return;
    }
    
    if (!isValidYouTubeUrl(url)) {
        showError('Please enter a valid YouTube URL');
        return;
    }
    
    showLoading();
    
    try {
        const formData = new FormData();
        formData.append('video_url', url);
        formData.append('summary_type', summaryType);
        formData.append('ai_provider', aiProvider);
        if (summaryType === 'bullet' && bulletCount) {
            formData.append('bullet_count', bulletCount);
        }
        
        const response = await fetch(`${API_BASE_URL}/summarize/youtube`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showResults(data);
        } else {
            showError(data.detail || 'An error occurred while summarizing the video');
        }
    } catch (error) {
        console.error('Error:', error);
        showError('Failed to connect to the server. Please make sure the backend is running.');
    }
}

// Summarize PDF document
async function summarizePDF() {
    const fileInput = document.getElementById('pdf-file');
    const file = fileInput.files[0];
    const summaryType = document.getElementById('summary-type').value;
    const aiProvider = 'gemini';
    const bulletCountEl = document.getElementById('bullet-count');
    const bulletCount = bulletCountEl ? bulletCountEl.value : '';
    
    // Validation
    if (!file) {
        showError('Please select a PDF file');
        return;
    }
    
    if (file.type !== 'application/pdf') {
        showError('Please select a valid PDF file');
        return;
    }
    
    // No client-side size cap; allow large PDFs as requested
    
    showLoading();
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('summary_type', summaryType);
        formData.append('ai_provider', aiProvider);
        if (summaryType === 'bullet' && bulletCount) {
            formData.append('bullet_count', bulletCount);
        }
        
        const response = await fetch(`${API_BASE_URL}/summarize/pdf`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showResults(data);
        } else {
            showError(data.detail || 'An error occurred while summarizing the PDF');
        }
    } catch (error) {
        console.error('Error:', error);
        showError('Failed to connect to the server. Please make sure the backend is running.');
    }
}



// ----- Download PDF (html2pdf first, fallback to backend) ----- //
function downloadSummaryPDF() {
    const source = document.getElementById("summary-text");
    if (!source || !source.innerHTML.trim()) {
        showError("Generate a summary first.");
        return;
    }

    // Clone the summary so we can cleanly print it
    const clone = source.cloneNode(true);

    // Ensure bullet dots
    clone.querySelectorAll("li").forEach(li => {
        let t = li.innerText.trim();
        if (!t.startsWith("•")) li.innerText = "• " + t;
    });

    // Open a new print window
    const printWindow = window.open("", "_blank");

    printWindow.document.write(`
        <html>
        <head>
            <title>Summary PDF</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    padding: 25px;
                    line-height: 1.6;
                    color: #000;
                }
                h3, h4 {
                    margin-top: 20px;
                    margin-bottom: 8px;
                    font-weight: bold;
                    border-bottom: 1px solid #aaa;
                    padding-bottom: 4px;
                }
                p {
                    margin-bottom: 12px;
                }
                ul {
                    margin-bottom: 12px;
                    padding-left: 22px;
                }
                li {
                    margin-bottom: 6px;
                }
            </style>
        </head>
        <body>
            ${clone.innerHTML}
        </body>
        </html>
    `);

    printWindow.document.close();
    printWindow.focus();

    // Trigger browser PDF exporter
    printWindow.print();

    // Close the print window after printing
    printWindow.onafterprint = () => {
        printWindow.close();
    };
}


// Check API health
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        
        if (response.ok) {
            console.log('API is healthy:', data);
        } else {
            console.warn('API health check failed:', data);
        }
    } catch (error) {
        console.error('API health check failed:', error);
    }
}

// Check API health on page load
document.addEventListener('DOMContentLoaded', function() {
    checkAPIHealth();
});
