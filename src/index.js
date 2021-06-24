import './css/style.css'
import './css/simple-grid.css'

import * as images from './js/images';

import * as tf from '@tensorflow/tfjs';
import { ResizeLayer } from './js/layers';
import { recolor, rgbToHsv } from './js/recoloring';

const NN_SIZE = [128, 128];
const ORIG_SIZE = [512, 512];

const webcamButton = document.getElementById('webcamButton');
const video = document.getElementById('webcam');
const canvasElement = document.getElementById('canvas');
const modelButton = document.getElementById('modelButton');
const fps = document.getElementById('fps');
var model = undefined;
var webcam = undefined;
var videoSourcesSelect = undefined;

navigator.mediaDevices.enumerateDevices().then((devices) => {
  videoSourcesSelect = document.getElementById("video-source");

  devices.forEach((device) => {
      let option = new Option();
      option.value = device.deviceId;

      switch(device.kind){
          case "videoinput":
              option.text = device.label || `Camera ${videoSourcesSelect.length + 1}`;
              videoSourcesSelect.appendChild(option);
              break;
      }
  });
}).catch(function (e) {
  console.log(e.name + ": " + e.message);
});

var imageTensor = undefined;
const image = new Image();
image.crossOrigin = 'anonymous';
image.src = images.textures[0];
image.onload = () => {
  tf.tidy(() => {
    imageTensor = tf.browser.fromPixels(image);
    imageTensor = tf.image.resizeBilinear(imageTensor, ORIG_SIZE).arraySync();
    for (let i = 0; i < ORIG_SIZE[0]; ++i) {
      for (let j = 0; j < ORIG_SIZE[1]; ++j) {
        imageTensor[i][j] = rgbToHsv(imageTensor[i][j][0], imageTensor[i][j][1], imageTensor[i][j][2]);
      }
    }
  });
}

modelButton.addEventListener('click', modelButtonClicked);

async function modelButtonClicked() {
  tf.serialization.registerClass(ResizeLayer);
  model = await tf.loadLayersModel("model.json");

  modelButton.classList.add("removed");

  webcamButton.classList.remove("removed");
  document.getElementById("video-source-label").classList.remove("removed");
  document.getElementById("video-source").classList.remove("removed");
}

webcamButton.addEventListener('click', webcamButtonClicked);

function webcamButtonClicked() {
  document.getElementById("camera-icon").classList.add("removed");
  video.classList.remove("removed");

  document.getElementById("sofa-icon").classList.add("removed");
  canvasElement.classList.remove("removed");
  fps.classList.remove("removed");

  webcamButton.classList.add("removed");
  document.getElementById("video-source-label").classList.add("removed");
  document.getElementById("video-source").classList.add("removed");

  const videoSource = videoSourcesSelect.value;
  const constraints = {
    video: {
      width: ORIG_SIZE[0],
      height: ORIG_SIZE[1],
      deviceId: videoSource ? {exact: videoSource} : undefined
    }
  };

  navigator.mediaDevices.getUserMedia(constraints).then(async function(stream) {
    video.srcObject = stream;
    video.addEventListener('loadeddata', predictWebcam);
    webcam = await tf.data.webcam(video, {
      facingMode: "environment",
      deviceId: videoSource ? videoSource : undefined
    });
  });
}

async function predictWebcam() {
  if (webcam === undefined) {
    return;
  }

  const start = performance.now();

  const original = await webcam.capture();
  const img = tf.image.resizeBilinear(original, ORIG_SIZE);

  const predictions = tf.tidy(() => {
    const preds = tf.image.resizeBilinear(img, NN_SIZE).reshape([1, NN_SIZE[0], NN_SIZE[1], 3]);
    return tf.image.resizeBilinear(model.predict(preds), ORIG_SIZE);

    // const preds = tf.add(tf.mul(tf.cast(img, 'float32'), 2 / 255), -1);
    // const preds = tf.div(tf.cast(img, 'float32'), 255);
  });

  const imgArr = await img.array();
  const predArr = await predictions.array();
  const recolored = await recolor(imgArr, predArr, imageTensor, ORIG_SIZE);

  tf.tidy(() => {
    tf.browser.toPixels(tf.tensor(recolored), canvasElement);
  });

  original.dispose();
  img.dispose();
  predictions.dispose();

  fps.textContent = "Elapsed time: " + (performance.now() - start) / 1000 + " seconds";

  window.requestAnimationFrame(predictWebcam);
}