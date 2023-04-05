// Get a reference to the video element and the label element
        const label = document.getElementById('label');
        const video = document.getElementById('videoStream');
        const button = document.getElementById('add-btn');
        const text = document.getElementById("get-label")
        const txtarea = document.getElementById("txt");

        let sentense = [];
        let letters = [];
        let letter = "";

        // Set the interval time (in milliseconds)
        const intervalTime = 1000;

        // Define the endpoint URL
        const endpointUrl = 'http://127.0.0.1:8080/label';
        const videoUrl = 'http://127.0.0.1:8080/video_feed';

        function getValueFromEndpoint(buttonClicked=false) {
          if(buttonClicked){
            fetch(endpointUrl)
            .then(response => response.json())
            .then( data => {
              label.innerText = `Label: ${data}`;
              console.log(data);
              creatSentense(data)
            })
            // , creatSentense(letter)
            .catch(error => {
              // Handle any errors that occur during the request
              console.error('Error retrieving value:', error);
            });
          }else{
            fetch(endpointUrl)
            .then(response => response.json())
            .then( data => {
              // Do something with the retrieved value
              label.innerText = `Label: ${data}`;
            })
            // , creatSentense(letter)
            .catch(error => {
              // Handle any errors that occur during the request
              console.error('Error retrieving value:', error);
            });
          }}

          function deletefunc() {
              text.value = '';
              sentense = []
              txtarea.value = ''
          }

          function creatSentense(label) {
            // console.log(label);
            sentense.push(label)
            console.log(sentense);
            text.value = sentense.join('')
            txtarea.value = sentense.join('')
          }

          button.addEventListener('click', function() {
            getValueFromEndpoint(true)
          });
// Call the function initially to retrieve the value immediately
getValueFromEndpoint();

// Set the interval to retrieve the value periodically
setInterval(getValueFromEndpoint, intervalTime);