//const mpHands = window;
//const drawingUtils = window;
//const controls = window;
//const controls3d = window;
//
//const videoElement = document.getElementsByClassName('input_video')[0];
//const canvasElement = document.getElementsByClassName('output_canvas')[0];
//const controlsElement = document.getElementsByClassName('control-panel')[0];
//const canvasCtx = canvasElement.getContext('2d');
//const config = { locateFile: (file) => {
//        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands@${mpHands.VERSION}/${file}`;
//    } };
//
//const fpsControl = new controls.FPS();
//// Optimization: Turn off animated spinner after its hiding animation is done.
//const spinner = document.querySelector('.loading');
//spinner.ontransitionend = () => {
//    spinner.style.display = 'none';
//};
//const landmarkContainer = document.getElementsByClassName('landmark-grid-container')[0];
//const grid = new controls3d.LandmarkGrid(landmarkContainer, {
//    connectionColor: 0xCCCCCC,
//    definedColors: [{ name: 'Left', value: 0xffa500 }, { name: 'Right', value: 0x00ffff }],
//    range: 0.2,
//    fitToGrid: false,
//    labelSuffix: 'm',
//    landmarkSize: 2,
//    numCellsPerAxis: 4,
//    showHidden: false,
//    centered: false,
//});
// function onResults(results) {
//    // Hide the spinner.
//    document.body.classList.add('loaded');
//    // Update the frame rate.
//    fpsControl.tick();
//    // Draw the overlays.
//    canvasCtx.save();
//    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
//    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
//    if (results.multiHandLandmarks && results.multiHandedness) {
//        for (let index = 0; index < results.multiHandLandmarks.length; index++) {
//            const classification = results.multiHandedness[index];
//            const isRightHand = classification.label === 'Right';
//            const landmarks = results.multiHandLandmarks[index];
//            drawingUtils.drawConnectors(canvasCtx, landmarks, mpHands.HAND_CONNECTIONS, { color: isRightHand ? '#00FF00' : '#FF0000' });
//            drawingUtils.drawLandmarks(canvasCtx, landmarks, {
//                color: isRightHand ? '#00FF00' : '#FF0000',
//                fillColor: isRightHand ? '#FF0000' : '#00FF00',
//                radius: (data) => {
//                    return drawingUtils.lerp(data.from.z, -0.15, .1, 10, 1);
//                }
//            });
//        }
//    }
//    canvasCtx.restore();
//    if (results.multiHandWorldLandmarks) {
//        // We only get to call updateLandmarks once, so we need to cook the data to
//        // fit. The landmarks just merge, but the connections need to be offset.
//        const landmarks = results.multiHandWorldLandmarks.reduce((prev, current) => [...prev, ...current], []);
//        const colors = [];``
//        let connections = [];
//        for (let loop = 0; loop < results.multiHandWorldLandmarks.length; ++loop) {
//            const offset = loop * mpHands.HAND_CONNECTIONS.length;
//            const offsetConnections = mpHands.HAND_CONNECTIONS.map((connection) => [connection[0] + offset, connection[1] + offset]);
//            connections = connections.concat(offsetConnections);
//            const classification = results.multiHandedness[loop];
//            colors.push({
//                list: offsetConnections.map((unused, i) => i + offset),
//                color: classification.label,
//            });
//        }
//        grid.updateLandmarks(landmarks, connections, colors);
//    }
//    else {
//        grid.updateLandmarks([]);
//    }
// }
//const hands = new mpHands.Hands(config);
//hands.onResults(onResults);
//// Present a control panel through which the user can manipulate the solution
//// options.
//new controls
//    .ControlPanel(controlsElement, {
//    selfieMode: true,
//    maxNumHands: 2,
//    modelComplexity: 1,
//    minDetectionConfidence: 0.5,
//    minTrackingConfidence: 0.5
//})
//    .add([
//    new controls.StaticText({ title: 'MediaPipe Hands' }),
//    fpsControl,
//    new controls.Toggle({ title: 'Selfie Mode', field: 'selfieMode' }),
//    new controls.SourcePicker({
//        onFrame: async (input, size) => {
//            const aspect = size.height / size.width;
//            let width, height;
//            if (window.innerWidth > window.innerHeight) {
//                height = window.innerHeight;
//                width = height / aspect;
//            }
//            else {
//                width = window.innerWidth;
//                height = width * aspect;
//            }
//            canvasElement.width = width;
//            canvasElement.height = height;
//            await hands.send({ image: input });
//        },
//    }),
//    new controls.Slider({
//        title: 'Max Number of Hands',
//        field: 'maxNumHands',
//        range: [1, 4],
//        step: 1
//    }),
//    new controls.Slider({
//        title: 'Model Complexity',
//        field: 'modelComplexity',
//        discrete: ['Lite', 'Full'],
//    }),
//    new controls.Slider({
//        title: 'Min Detection Confidence',
//        field: 'minDetectionConfidence',
//        range: [0, 1],
//        step: 0.01
//    }),
//    new controls.Slider({
//        title: 'Min Tracking Confidence',
//        field: 'minTrackingConfidence',
//        range: [0, 1],
//        step: 0.01
//    }),
//])
//    .on(x => {
//    const options = x;
//    videoElement.classList.toggle('selfie', options.selfieMode);
//    hands.setOptions(options);
//});

// Adapted from https://docs.opencv.org/3.4/dd/d00/tutorial_js_video_display.html
// import cv from "opencv.js";

// const config = { locateFile: (file) => {
//        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands@${mpHands.VERSION}/${file}`;
//    } };


let video = document.getElementById("videoInput"); // video is the id of video tag
const canvasElement = document.getElementById('canvasOutput');
const canvasCtx = canvasElement.getContext('2d');
video.width = 640;
video.height = 480;

navigator.mediaDevices
  .getUserMedia({ video: true, audio: false })
  .then(function (stream) {
    video.srcObject = stream;
    video.play();

    let src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
    let dst = new cv.Mat(video.height, video.width, cv.CV_8UC1);
    let cap = new cv.VideoCapture(video);
    

      const FPS = 30;
        function processVideo() {
          try {
              console.log(cap)

               const xhr = new XMLHttpRequest();
                xhr.open('POST', 'http://localhost:5000/video');
                xhr.setRequestHeader('Content-Type', 'application/json');
                xhr.send(JSON.stringify({cap: cap}));

           setTimeout(processVideo, 1000);
          } catch (err) {
            console.error(err);
          }
        }
       setTimeout(processVideo, 0);
      })
      .catch(function (err) {
        console.log("An error occurred! " + err);
      });
// let videoElement = document.getElementById('videoInput');
// const canvasElement = document.getElementById('canvasOutput');
// const canvasCtx = canvasElement.getContext('2d');
//
//
//         function onResults(results) {
//           canvasCtx.save();
//           canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
//           canvasCtx.drawImage(
//               results.image, 0, 0, canvasElement.width, canvasElement.height);
//           if (results.multiHandLandmarks) {
//             for (const landmarks of results.multiHandLandmarks) {
//               drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS,
//                              {color: '#00FF00', lineWidth: 5});
//               drawLandmarks(canvasCtx, landmarks, {color: '#FF0000', lineWidth: 2});
//             }
//           }
//           canvasCtx.restore();
//         }
        
