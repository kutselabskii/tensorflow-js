import * as tf from '@tensorflow/tfjs';

import logMessage from './js/logger'
import './css/style.css'

// import model from './models/model.json'

const enableWebcamButton = document.getElementById('webcamButton');
const video = document.getElementById('webcam');
const liveView = document.getElementById('liveView');
const demosSection = document.getElementById('demos');

const modelButton = document.getElementById('modelButton');
modelButton.addEventListener('click', modelButtonClicked);

async function modelButtonClicked() {
  model = await tf.loadLayersModel("model.json");
  demosSection.classList.remove('invisible');
}

var model = undefined
var children = [];

var webcam = undefined

enableWebcamButton.addEventListener('click', buttonClicked);

function buttonClicked(event) {
  event.target.classList.add('removed');  

  const constraints = {
    video: true
  };

  navigator.mediaDevices.getUserMedia(constraints).then(async function(stream) {
    video.srcObject = stream;
    video.addEventListener('loadeddata', predictWebcam);

    webcam = await tf.data.webcam(video, {
      resizeWidth: 196,
      resizeHeight: 196,
    });
  });
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

async function predictWebcam() {
  var img = await webcam.capture();
  img = img.reshape([1, 196, 196, 3]);

  var predictions = model.predict(img);
  console.log(indexOfMax(predictions));

  img.dispose();
    
    // Call this function again to keep predicting when the browser is ready.
    window.requestAnimationFrame(predictWebcam);
}