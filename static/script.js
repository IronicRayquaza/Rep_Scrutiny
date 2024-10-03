// Fetch video stream
document.getElementById('videoElement').src = "/video_feed"; // Use a relative path to your video feed

// Update status (reps, stage, prob) every second
setInterval(function () {
    fetch('/get_status')
        .then(response => response.json())
        .then(data => {
            document.getElementById('stage').innerText = data.stage;
            document.getElementById('reps').innerText = data.counter;
            document.getElementById('prob').innerText = data.probability.toFixed(2);
        });
}, 1000);
