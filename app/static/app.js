const video = document.getElementById("webcam");
const canvas = document.getElementById("canvas");
const capturedImage = document.getElementById("capturedImage");
const startCameraBtn = document.getElementById("startCamera");
const capturePhotoBtn = document.getElementById("capturePhoto");
const retakePhotoBtn = document.getElementById("retakePhoto");
const uploadLabel = document.getElementById("uploadLabel");
const fileUpload = document.getElementById("fileUpload");
const form = document.getElementById("registrationForm");
const resultBox = document.getElementById("resultBox");

let stream = null;
let photoBlob = null;

async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
    } catch (err) {
        showResult(`Could not access camera: ${err.message}. Use "Upload Photo Instead".`, false);
        return;
    }
    video.srcObject = stream;
    startCameraBtn.style.display = "none";
    capturePhotoBtn.style.display = "inline-flex";
    uploadLabel.style.display = "none";
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach((track) => track.stop());
        stream = null;
    }
}

function capturePhoto() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    canvas.toBlob((blob) => {
        photoBlob = blob;
        capturedImage.src = URL.createObjectURL(blob);
        capturedImage.style.display = "block";
        video.style.display = "none";
        stopCamera();
        capturePhotoBtn.style.display = "none";
        retakePhotoBtn.style.display = "inline-flex";
    }, "image/jpeg", 0.92);
}

function retakePhoto() {
    photoBlob = null;
    capturedImage.style.display = "none";
    video.style.display = "block";
    retakePhotoBtn.style.display = "none";
    uploadLabel.style.display = "inline-flex";
    fileUpload.value = "";
    startCamera();
}

function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    photoBlob = file;
    capturedImage.src = URL.createObjectURL(file);
    capturedImage.style.display = "block";
    video.style.display = "none";
    stopCamera();
    startCameraBtn.style.display = "none";
    capturePhotoBtn.style.display = "none";
    retakePhotoBtn.style.display = "inline-flex";
}

function showResult(message, ok) {
    resultBox.textContent = message;
    resultBox.className = `result-box ${ok ? "success" : "error"}`;
    resultBox.style.display = "block";
}

startCameraBtn.addEventListener("click", startCamera);
capturePhotoBtn.addEventListener("click", capturePhoto);
retakePhotoBtn.addEventListener("click", retakePhoto);
fileUpload.addEventListener("change", handleFileUpload);

form.addEventListener("reset", () => {
    photoBlob = null;
    stopCamera();
    capturedImage.style.display = "none";
    video.style.display = "block";
    startCameraBtn.style.display = "inline-flex";
    capturePhotoBtn.style.display = "none";
    retakePhotoBtn.style.display = "none";
    uploadLabel.style.display = "inline-flex";
    resultBox.style.display = "none";
});

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    if (!photoBlob) {
        showResult("Please capture or upload a photo before submitting.", false);
        return;
    }

    const formData = new FormData();
    formData.append("full_name", document.getElementById("fullName").value);
    formData.append("image", photoBlob, "photo.jpg");

    const submitBtn = document.getElementById("submitBtn");
    submitBtn.disabled = true;
    submitBtn.textContent = "Registering...";

    try {
        const response = await fetch("/api/v1/registrations/register", {
            method: "POST",
            body: formData,
        });
        const data = await response.json();
        if (response.ok) {
            showResult(`Registration successful.\nID: ${data.registration_id}\nStatus: ${data.status}`, true);
        } else {
            showResult(`Registration failed: ${data.detail || "unknown error"}`, false);
        }
    } catch (err) {
        showResult(`Request failed: ${err.message}`, false);
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = "Register";
    }
});
