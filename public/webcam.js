const button = document.getElementById('webcamButton');
button.addEventListener('click', buttonClicked);

function buttonClicked() {
    var video = document.getElementById('webcam');
    window.navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
          video.srcObject = stream;
          video.onloadedmetadata = (e) => {
              video.play();
          };
      })
      .catch( () => {
          alert('You have give browser the permission to run Webcam and mic ;( ');
      });
}