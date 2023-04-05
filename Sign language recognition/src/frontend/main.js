// Get a reference to the video element and the label element
        const label = document.getElementById('label');
        const video = document.getElementById('videoStream');

        // Set the interval time (in milliseconds)
        const intervalTime = 1000;

        // Define the endpoint URL
        const endpointUrl = 'http://127.0.0.1:8080/label';
        const videoUrl = 'http://127.0.0.1:8080/video_feed';

        // function getVideoFromEndpoint() {
        //   fetch(videoUrl)
        //   .then(response => response.json())
        //   .then(data => {
        //     video.src
        //   })
        // }

        function getValueFromEndpoint() {
          fetch(endpointUrl)
            .then(response => response.json())
            .then(data => {
              // Do something with the retrieved value
              label.innerText = `Label: ${data}`;
              console.log('Retrieved value:', data);
            })
            .catch(error => {
              // Handle any errors that occur during the request
              console.error('Error retrieving value:', error);
            });
          }

// Call the function initially to retrieve the value immediately
getValueFromEndpoint();

// Set the interval to retrieve the value periodically
setInterval(getValueFromEndpoint, intervalTime);