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


const enableWebcamButton = document.getElementById('webcamButton');
const video = document.getElementById('webcam');
const canvasElement = document.getElementsByClassName('output_canvas')[0];
const canvasCtx = canvasElement.getContext('2d');
const demosSection = document.getElementById('demos');
const modelButton = document.getElementById('modelButton');
var model = undefined;
var webcam = undefined

var imageTensor = undefined;
const image = new Image();
image.crossOrigin = 'anonymous';
image.src = i1;
image.onload = () => {
  imageTensor = tf.browser.fromPixels(image);
  imageTensor = tf.image.resizeBilinear(imageTensor, [512, 512]).arraySync();
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

  const constraints = {
    video: {
      width: 512,
      height: 512,
      facingMode: 'environment'
    }
  };
  navigator.mediaDevices.getUserMedia(constraints).then(async function(stream) {
    video.srcObject = stream;
    video.addEventListener('loadeddata', predictWebcam);

    webcam = await tf.data.webcam(video, {
      resizeWidth: 512,
      resizeHeight: 512,
    });
  });
}

function recolor(image, mask) {
  for (let i = 0; i < 512; ++i) {
    for (let j = 0; j < 512; ++j) {
      if (true || mask[0][i][j][1] > 0.5) {
        image[0][i][j] = [imageTensor[i][j][0] / 255, imageTensor[i][j][1] / 255, imageTensor[i][j][2] / 255];
      } else {
        const img = image[0][i][j];
        image[0][i][j] = [img[0] / 255, img[1] / 255, img[2] / 255];
      }
    }
  }
  return image;
}

async function predictWebcam() {
  if (webcam === undefined) {
    return;
  }

  var img = await webcam.capture();
  img = img.reshape([1, 512, 512, 3]);
  var predictions = model.predict(img);

  const recolored = recolor(img.arraySync(), predictions.arraySync());
  // canvasCtx.drawImage(recolored, 0, 0, width, height);

  // console.log(recolored);

  const drawer = tf.squeeze(tf.tensor(recolored));
  // console.log(drawer);
  tf.browser.toPixels(drawer, canvasElement);

  window.requestAnimationFrame(predictWebcam);

  img.dispose();
} 