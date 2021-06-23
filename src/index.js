import * as tf from '@tensorflow/tfjs';
import './css/style.css'

import i1 from './img/1.jpg';

class ResizeLayer extends tf.layers.Layer {
    constructor(config) {
        super(config);
        this.w = config.w;
        this.h = config.h;
      }
  
    call(input) {
      return tf.tidy(() => {
        return tf.image.resizeBilinear(input[0], [this.w, this.h]);
      });
      }
  
    computeOutputShape(input_shape) {
      return [input_shape[0], this.w, this.h, input_shape[3]]
    }
  
    static get className() {
      return 'ResizeLayer';
    }
  }

const NN_SIZE = [128, 128];
const ORIG_SIZE = [512, 512];

const enableWebcamButton = document.getElementById('webcamButton');
const video = document.getElementById('webcam');
const canvasElement = document.getElementsByClassName('output_canvas')[0];
const canvasCtx = canvasElement.getContext('2d');
const demosSection = document.getElementById('demos');
const modelButton = document.getElementById('modelButton');
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
image.src = i1;
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
  modelButton.setAttribute('display', 'none');
  demosSection.classList.remove('invisible');
}

enableWebcamButton.addEventListener('click', buttonClicked);

function buttonClicked(event) {
  event.target.classList.add('removed');
  enableWebcamButton.setAttribute('display', 'none');
  const videoSource = videoSourcesSelect.value;
  const constraints = {
    video: {
      width: ORIG_SIZE[0],
      height: ORIG_SIZE[1],
      deviceId: videoSource ? {exact: videoSource} : undefined
      // facingMode: 'environment'
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

 function rgbToHsv(r, g, b) {
  r /= 255, g /= 255, b /= 255;

  var max = Math.max(r, g, b), min = Math.min(r, g, b);
  var h, s, v = max;

  var d = max - min;
  s = max == 0 ? 0 : d / max;

  if (max == min) {
    h = 0; // achromatic
  } else {
    switch (max) {
      case r: h = (g - b) / d + (g < b ? 6 : 0); break;
      case g: h = (b - r) / d + 2; break;
      case b: h = (r - g) / d + 4; break;
    }

    h /= 6;
  }

  return [ h, s, v ];
}

function hsvToRgb(h, s, v) {
  var r, g, b;

  var i = Math.floor(h * 6);
  var f = h * 6 - i;
  var p = v * (1 - s);
  var q = v * (1 - f * s);
  var t = v * (1 - (1 - f) * s);

  switch (i % 6) {
    case 0: r = v, g = t, b = p; break;
    case 1: r = q, g = v, b = p; break;
    case 2: r = p, g = v, b = t; break;
    case 3: r = p, g = q, b = v; break;
    case 4: r = t, g = p, b = v; break;
    case 5: r = v, g = p, b = q; break;
  }

  return [r, g, b];
}

async function recolor(image, mask) {
  for (let i = 0; i < ORIG_SIZE[0]; ++i) {
    for (let j = 0; j < ORIG_SIZE[1]; ++j) {
      const img = image[i][j];
      
      if (mask[0][i][j][0] > 0.98) {
        const i_hsv = rgbToHsv(img[0], img[1], img[2]);
        const res_hsv = [imageTensor[i][j][0], imageTensor[i][j][1], i_hsv[2]];
        image[i][j] = hsvToRgb(res_hsv[0], res_hsv[1], res_hsv[2]);
      } else {
        image[i][j] = [img[0] / 255, img[1] / 255, img[2] / 255];
      }
    }
  }
  return image;
}

async function predictWebcam() {
  if (webcam === undefined) {
    return;
  }

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
  const recolored = await recolor(imgArr, predArr);

  tf.tidy(() => {
    tf.browser.toPixels(tf.tensor(recolored), canvasElement);
  });

  original.dispose();
  img.dispose();
  predictions.dispose();

  window.requestAnimationFrame(predictWebcam);
}