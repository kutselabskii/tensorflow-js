import './css/style.css'
import './css/simple-grid.css'

import * as tf from '@tensorflow/tfjs';

import * as images from './js/images';
import * as models from './js/models';

const size = [512, 512];

var webcamButton = undefined;
var video = undefined;
var canvasElement = undefined;
var modelButton = undefined;
var fps = undefined;
var webcam = undefined;
var videoSourcesSelect = undefined;
var modelSourcesSelect = undefined;
var currentImage = 0;
var working = false;
var fileReaderElement = undefined;
var customImage = undefined;

const availableModels = [new models.LinkNet(size), new models.FastSCNN(size), new models.UNet(size)];

initialize();

function initialize() {
  prepareElements();
  prepareVideoSources();
  prepareModelSources();
  prepareTypeSelector();
  prepareImageLoader();

  let textures = [];
  for (let i = 0; i < images.textures.length; ++i) {
    textures.push(document.getElementById("texture-" + i));
  }
  images.prepareTextureTensors(size, textures);
}

function prepareElements() {
  webcamButton = document.getElementById('webcamButton');
  video = document.getElementById('webcam');
  canvasElement = document.getElementById('canvas');
  modelButton = document.getElementById('modelButton');
  fps = document.getElementById('fps');
  videoSourcesSelect = document.getElementById("video-source");
  modelSourcesSelect = document.getElementById("model-source");

  prepareImages();
}

function prepareImages() {
  const empty = document.getElementById('empty');
  for (let i = 0; i < images.textures.length; ++i) {
    const enclosedIndex = i;
    const img = document.getElementById('texture-' + i);

    img.addEventListener('click', () => {
      if (enclosedIndex == currentImage) {
        return;
      }

      if (currentImage != -1) {
        document.getElementById('texture-' + currentImage).classList.remove('highlight');
      } else {
        empty.classList.remove('highlight');

        if (working) {
          video.classList.add('removed');
          canvasElement.classList.remove('removed');
        } else {
          document.getElementById("camera-icon").classList.add("removed");
          document.getElementById("sofa-icon").classList.remove("removed");
        }
      }
      img.classList.add('highlight');
      currentImage = enclosedIndex;
    });
  }

  empty.addEventListener('click', () => {
    if (currentImage == -1) {
      return;
    }

    if (working) {
      if (customImage === undefined) {
        video.classList.remove('removed');
        canvasElement.classList.add('removed');
      }
    } else {
        document.getElementById("camera-icon").classList.remove("removed");
        document.getElementById("sofa-icon").classList.add("removed");
    }
    document.getElementById('texture-' + currentImage).classList.remove('highlight');
    empty.classList.add('highlight');
    currentImage = -1;
  })
}

function prepareModelSources() {
  for (let i = 0; i < availableModels.length; ++i) {
    let option = new Option();
    option.value = i;
    option.text = availableModels[i].name;
    modelSourcesSelect.appendChild(option);
  }
}

function prepareTypeSelector() {
  document.getElementById("typeWebcamButton").addEventListener("click", () => {
    document.getElementById("type-selection-div").classList.add("removed");
    document.getElementById("video-settings").classList.remove("removed");
  });

  document.getElementById("typeLoadButton").addEventListener("click", () => {
    document.getElementById("type-selection-div").classList.add("removed");
    document.getElementById("camera-icon").classList.add("removed");
    document.getElementById("sofa-icon").classList.add("removed");
    document.getElementById("file-loader-div").classList.remove("removed");
    document.getElementById("fps-div").classList.remove("removed");

    video.classList.add("removed");
    canvasElement.classList.remove("removed");

    availableModels[modelSourcesSelect.value].setScreens(undefined, canvasElement);
    working = true;
  });
}

function prepareVideoSources() {
  navigator.mediaDevices.enumerateDevices().then((devices) => {
    devices.forEach((device) => {
        let option = new Option();
        option.value = device.deviceId;

        if (device.kind == "videoinput") {
          option.text = device.label || `Camera ${videoSourcesSelect.length + 1}`;
          videoSourcesSelect.appendChild(option);  
        }
    });
  }).catch(function (e) {
    console.log(e.name + ": " + e.message);
  });
}

function prepareImageLoader() {
  fileReaderElement = document.getElementById("file-loader");
  fileReaderElement.addEventListener("change", () => {
      var reader = new FileReader();
      reader.onload = function(e) {
        var image = new Image();
        image.crossOrigin = 'anonymous';

        image.onload = async () => {
          if (customImage !== undefined) {
            customImage.dispose();
          }
          
          customImage = tf.image.resizeBilinear(tf.browser.fromPixels(image), size);
          await predictWebcam();
        };
    
        image.src = e.target.result;
      };
      reader.readAsDataURL(fileReaderElement.files[0]);
    }
  );
}

modelButton.addEventListener('click', modelButtonClicked);
async function modelButtonClicked() {
  document.getElementById("model-settings").classList.add("removed");
  document.getElementById("loader").classList.remove("removed");

  await availableModels[modelSourcesSelect.value].load();

  document.getElementById("loader").classList.add("removed");
  document.getElementById("type-selection-div").classList.remove("removed");
}

webcamButton.addEventListener('click', webcamButtonClicked);
function webcamButtonClicked() {
  working = true;

  document.getElementById("camera-icon").classList.add("removed");
  document.getElementById("sofa-icon").classList.add("removed");

  if (currentImage == -1) {
    video.classList.remove("removed");
  } else {
    canvasElement.classList.remove("removed");
  }

  document.getElementById("fps-div").classList.remove("removed");
  document.getElementById("video-settings").classList.add("removed");

  const videoSource = videoSourcesSelect.value;
  const constraints = {
    video: {
      width: size[0],
      height: size[1],
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
    availableModels[modelSourcesSelect.value].setScreens(webcam, canvasElement);
  });
}

async function predictWebcam() {
  var time = undefined;
  if (customImage === undefined) {
    time = await availableModels[modelSourcesSelect.value].predict(currentImage == -1 ? undefined : images.tensors[currentImage], undefined);
  } else {
    if (currentImage == -1) {
      tf.browser.toPixels(customImage.div(255), canvasElement);
      time = 0;
    } else {
      time = await availableModels[modelSourcesSelect.value].predict(images.tensors[currentImage], tf.clone(customImage));
    }
  }

  if (time < 0) {
    return;
  }

  fps.textContent = "Elapsed time: " + (time / 1000) + " seconds";

  window.requestAnimationFrame(predictWebcam);
}