const statusDiv = document.getElementById('status');

function setStatus(message, isError = false) {
  statusDiv.textContent = message;
  statusDiv.className = isError ? 'error' : '';
}

async function sendRegistration(name, imageData) {
  try {
    const response = await fetch('http://localhost:5000/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, image: imageData }),
    });
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    const data = await response.json();
    setStatus(data.message, data.status !== 'success');
  } catch (error) {
    console.error('Fetch failed:', error);
    setStatus('Failed to send registration request.', true);
  }
}

function registerFromUpload() {
  const name = document.getElementById('uploadName').value.trim();
  const fileInput = document.getElementById('upload');
  const file = fileInput.files[0];

  if (!name || !file) {
    setStatus('Please provide a name and select an image.', true);
    return;
  }

  const reader = new FileReader();
  reader.onload = function (e) {
    sendRegistration(name, e.target.result);
  };
  reader.onerror = function () {
    setStatus('Failed to read the image file.', true);
  };
  reader.readAsDataURL(file);
}

function registerFromLive() {
  const name = document.getElementById('name').value.trim();
  const video = document.getElementById('live-feed');
  const canvas = document.getElementById('snapshot');
  const context = canvas.getContext('2d');

  if (!name) {
    setStatus('Please provide a name.', true);
    return;
  }

  if (!video || !video.complete || !video.naturalWidth) {
    setStatus('Webcam feed not ready.', true);
    return;
  }

  canvas.width = video.naturalWidth;
  canvas.height = video.naturalHeight;
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  const imageData = canvas.toDataURL('image/jpeg', 0.8);
  sendRegistration(name, imageData);
}