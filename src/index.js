// import * as tf from '@tensorflow/tfjs';
// import './css/style.css'

// const enableWebcamButton = document.getElementById('webcamButton');
// const video = document.getElementById('webcam');
// const demosSection = document.getElementById('demos');
// const modelButton = document.getElementById('modelButton');
// var model = undefined;
// var webcam = undefined

// modelButton.addEventListener('click', modelButtonClicked);

// async function modelButtonClicked() {
//   model = await tf.loadLayersModel("model.json");
//   modelButton.setAttribute('display', 'none');
//   demosSection.classList.remove('invisible');
// }

// enableWebcamButton.addEventListener('click', buttonClicked);

// function buttonClicked(event) {
//   event.target.classList.add('removed');
//   enableWebcamButton.setAttribute('display', 'none');

//   const constraints = {
//     video: true
//   };
//   navigator.mediaDevices.getUserMedia(constraints).then(async function(stream) {
//     video.srcObject = stream;
//     video.addEventListener('loadeddata', predictWebcam);

//     webcam = await tf.data.webcam(video, {
//       resizeWidth: 196,
//       resizeHeight: 196,
//     });
//   });
// }

// function indexOfMax(arr) {
//   if (arr.length === 0) {
//       return -1;
//   }
//   var max = arr[0];
//   var maxIndex = 0;
//   for (var i = 1; i < arr.length; i++) {
//       if (arr[i] > max) {
//           maxIndex = i;
//           max = arr[i];
//       }
//   }
//   return maxIndex;
// }

// async function predictWebcam() {
//   // var img = await webcam.capture();
//   // img = img.reshape([1, 196, 196, 3]);

//   // var predictions = model.predict(img);
//   // console.log(indexOfMax(predictions));

//     // Call this function again to keep predicting when the browser is ready.
//     window.requestAnimationFrame(predictWebcam);
// } 