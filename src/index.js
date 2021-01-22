import * as tf from '@tensorflow/tfjs';
import './css/style.css'

import i1 from './img/1.png';
import i2 from './img/2.png';
import i3 from './img/3.png';
import i4 from './img/4.png';
import i5 from './img/5.png';
import i6 from './img/6.png';
import i7 from './img/7.png';
import i8 from './img/8.png';
import i9 from './img/9.png';
import i10 from './img/10.png';
import i11 from './img/11.png';

const images = [
  i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11
]

const enableWebcamButton = document.getElementById('webcamButton');
const video = document.getElementById('webcam');
const demosSection = document.getElementById('demos');
const canvasElement = document.getElementsByClassName('output_canvas')[0];
const canvasCtx = canvasElement.getContext('2d');
const modelButton = document.getElementById('modelButton');
const imgBlock = document.getElementById('images');
var last = 0;
var model = undefined;
var imgNodes = [];

for (let i = 0; i < 11; ++i) {
  var im = document.createElement('img');
  im.src = images[i];
  im.classList.add('img');
  imgBlock.appendChild(im);
  imgNodes.push(im);
}

const camera = new Camera(video, {
  onFrame: async () => {
    await hands.send({image: video});
  },
  width: 1280,
  height: 960
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
    var res = indexOfMax(predictions.arraySync()[0]);
    if (res != last) {
      imgNodes[last].classList.remove('chosen');
      imgNodes[res].classList.add('chosen');
    }
    last = res;
    // console.log(indexOfMax(predictions.arraySync()[0]));
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
  modelButton.setAttribute('display', 'none');
  demosSection.classList.remove('invisible');
}

enableWebcamButton.addEventListener('click', buttonClicked);

function buttonClicked(event) {
  event.target.classList.add('removed');
  // video.classList.add('removed');
  camera.start();
  enableWebcamButton.setAttribute('display', 'none');
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