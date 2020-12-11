let constraints;
let imageCapture;
let mediaStream

const img = document.querySelector('#photo');
const video = document.querySelector('#video');
const startButton = document.querySelector('button');

(async function () {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const camera = devices.filter(device => device.kind === 'videoinput')[0];

    if (mediaStream) {
        mediaStream.getTracks().forEach(track => {
            track.stop();
        });
    }

    constraints = {
        video: {deviceId: {exact: camera.deviceId}}
    };

    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    mediaStream = stream;
    video.srcObject = stream;
    imageCapture = new ImageCapture(stream.getVideoTracks()[0]);
    console.log(imageCapture)
})();

startButton.onclick = takePhoto

async function takePhoto() {
    const blob = await imageCapture.takePhoto();
    
    const reader = new FileReader();

    reader.onload = () => {
        const dataUrl = reader.result;
        const base64 = dataUrl.split(',')[1];

        fetch("/verify", { 
            method: "POST", 
            body: JSON.stringify({img: base64}),
            headers: {
                "Content-Type": "application/json"
            }
        })
        
    }
    reader.readAsDataURL(blob)
}