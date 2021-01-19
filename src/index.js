import * as tf from '@tensorflow/tfjs';
import './css/style.css'

const enableWebcamButton = document.getElementById('webcamButton');
const video = document.getElementById('webcam');
const demosSection = document.getElementById('demos');
const canvasElement = document.getElementsByClassName('output_canvas')[0];
const canvasCtx = canvasElement.getContext('2d');
const modelButton = document.getElementById('modelButton');
var model = undefined;

const camera = new Camera(video, {
  onFrame: async () => {
    await hands.send({image: video});
  },
  width: 640,
  height: 480
});

function onResults(results) {
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.drawImage(
      results.image, 0, 0, canvasElement.width, canvasElement.height);
  if (results.multiHandLandmarks) {
    let data = [];
    for (const landmarks of results.multiHandLandmarks) {
      drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS,
                      {color: '#00FF00', lineWidth: 5});
      drawLandmarks(canvasCtx, landmarks, {color: '#FF0000', lineWidth: 2});
      for (const i in landmarks) {
        const mark = landmarks[i];
        data.push([mark.x, mark.y]);
      }
    }

    var predictions = model.predict(tf.tensor(data).reshape([1, 21, 2]));
    console.log(indexOfMax(predictions.arraySync()[0]));
  }
  canvasCtx.restore();
}

const hands = new Hands({locateFile: (file) => {
  return `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.1/${file}`;
}});

hands.setOptions({
  maxNumHands: 1,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});
hands.onResults(onResults);

modelButton.addEventListener('click', modelButtonClicked);

async function modelButtonClicked() {
  model = await tf.loadLayersModel("model.json");
  demosSection.classList.remove('invisible');
}

enableWebcamButton.addEventListener('click', buttonClicked);

function buttonClicked(event) {
  event.target.classList.add('removed');  
  camera.start();
}

function indexOfMax(arr) {
  if (arr.length === 0) {
      return -1;
  }
  var max = arr[0];
  var maxIndex = 0;
  for (var i = 1; i < arr.length; i++) {
      if (arr[i] > max) {
          maxIndex = i;
          max = arr[i];
      }
  }
  return maxIndex;
}