// Constants for API URLs
const TEST_IMAGE_URL = "http://127.0.0.1:8000/CNN_gen1_test";
const CORRECT_URL = "http://127.0.0.1:8000/CNN_gen1_correct";
const INCORRECT_URL = "http://127.0.0.1:8000/CNN_gen1_incorrect";
// DOM elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('imgInp');
const preview = document.getElementById('preview');
const bodyEl = document.querySelector('.body');
const resultPane = document.getElementById('resultPane');
const predictionLine = document.getElementById('predictionLine');
const appleLine = document.getElementById('appleLine');
const orangeLine = document.getElementById('orangeLine');

// Function to show the right panel
function showRightPanel() {
    if (bodyEl) bodyEl.classList.add('submitted');
    if (resultPane) resultPane.hidden = false;
}

// Function to hide the right panel and reset UI
function resetUI() {
    if (bodyEl) bodyEl.classList.remove('submitted');
    if (resultPane) resultPane.hidden = true;
    // Clear previous results
    predictionLine.textContent = "The prediction is :";
    appleLine.textContent = "Apple :";
    orangeLine.textContent = "Orange :";
}


let file_address,label
// Function to handle API call
async function apicall() {
    if (!fileInput.files || !fileInput.files[0]) {
        alert("Please select an image before submitting.");
        return;
    }
    //data made in multipart/form-data format
    const dataImage = fileInput.files[0];
    const formData = new FormData();
    formData.append("train_file", dataImage);
    console.log({ method: "POST", body: formData });
    
    try{
        const response=await fetch(TEST_IMAGE_URL,{method:"POST",body:formData});
        const jsonresponse=await response.json();
        
        predictionLine.textContent+=jsonresponse.prediction;
        appleLine.textContent+=jsonresponse.Apple;
        orangeLine.textContent+=jsonresponse.Orange;
    
        file_address=jsonresponse.File_addr;
        label=jsonresponse.prediction;
        showRightPanel();
    }
    catch (error) {
        alert(error)
    }
}

async function del_file_stored() {
    try {
        data={"file_addr": file_address};
        const response=await fetch(`${CORRECT_URL}?file_addr=${file_address}`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(data)}); //this is not json?
        const jsonresponse=await response.json();
        console.log(jsonresponse.message);
        resetUI();
        clearInput()
    }
    catch (error)
    {
        alert(error);
    }
}

async function retrain() {
    let updated_label;
    if (label=="Apple"){
            updated_label="Orange";
    }
    else {
        updated_label="Apple";
    }
    
    try {
        
        data={"file_addr":file_address, "label":updated_label }
        const response=await  fetch(`${INCORRECT_URL}?file_addr=${file_address}&label=${updated_label}`,{method:"POST"});
        const jsonresponse=await  response.json();
        console.log(jsonresponse.message);
        resetUI();
        clearInput()
    }
    catch (error)
    {
        alert(error)
    }
}

function clearInput() {
  fileInput.value = ''; // Sets the value to an empty string
  preview.src = '';
  preview.style.display = 'none';
}

// Function to load and preview the selected file
function loadFile(event) {
    const file = event.target.files[0];
    if (!file) return;

    resetUI();

    // Clear previous image
    preview.src = '';
    preview.style.display = 'none';

    // Show the new image
    preview.src = URL.createObjectURL(file);
    preview.style.display = 'block';

    preview.onload = function() {
        URL.revokeObjectURL(preview.src); // Free memory
    };
}

// Initialize drag and drop support
window.addEventListener('DOMContentLoaded', function () {
    if (!uploadArea || !fileInput) return;

    // Prevent double-open when clicking directly on the input
    fileInput.addEventListener('click', function (e) {
        e.stopPropagation();
    });

    // Click anywhere in the upload area to open file dialog
    uploadArea.addEventListener('click', function (e) {
        if (e.target === fileInput) return;
        fileInput.click();
    });

    // Prevent default browser behavior for drag events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(function (eventName) {
        uploadArea.addEventListener(eventName, function (e) {
            e.preventDefault();
            e.stopPropagation();
        });
    });

    // Add dragover class
    ['dragenter', 'dragover'].forEach(function (eventName) {
        uploadArea.addEventListener(eventName, function () {
            uploadArea.classList.add('dragover');
        });
    });

    // Remove dragover class
    ['dragleave', 'drop'].forEach(function (eventName) {
        uploadArea.addEventListener(eventName, function () {
            uploadArea.classList.remove('dragover');
        });
    });

    // Handle drop
    uploadArea.addEventListener('drop', function (e) {
        const dt = e.dataTransfer;
        if (!dt || !dt.files || !dt.files[0]) return;

        fileInput.files = dt.files;
        loadFile({ target: fileInput });
    });
});