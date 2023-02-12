// Automatic Slideshow - change image every 4 seconds
var myIndex = 0;
carousel();

var mydata = JSON.parse(data);
console.log(mydata)

function carousel() {
  var i;
  var x = document.getElementsByClassName("mySlides");
  for (i = 0; i < x.length; i++) {
    x[i].style.display = "none";  
  }
  myIndex++;
  if (myIndex > x.length) {myIndex = 1}    
  x[myIndex-1].style.display = "block";  
  setTimeout(carousel, 4000);    
}

// Used to toggle the menu on small screens when clicking on the menu button
function myFunction() {
  var x = document.getElementById("navDemo");
  if (x.className.indexOf("w3-show") == -1) {
    x.className += " w3-show";
  } else { 
    x.className = x.className.replace(" w3-show", "");
  }
}

// When the user clicks anywhere outside of the modal, close it
var modal = document.getElementById('ticketModal');
window.onclick = function(event) {
  if (event.target == modal) {
    modal.style.display = "none";
  }
}

document.getElementById("start").addEventListener("change", function() {
    var input = this.value;
    var dateEntered = new Date(input);
    console.log(input); //e.g. 2015-11-13
    const predictionArray = mydata.find(predict => predict.date === input);
    console.log(predictionArray);
    const h_actualDate = document.getElementById("actualDate");
    h_actualDate.innerHTML = input;
    const h_actualH = document.getElementById("actualH");
    h_actualH.innerHTML = predict.offset;
    const h_actualPastH = document.getElementById("actualPastH");
    h_actualPastH.innerHTML = predict.observed;
    const h_actualPredictedH = document.getElementById("actualPredictedH");
    h_actualPredictedH.innerHTML = predict.predicted;
    //console.log(mydata[0]);
    console.log(dateEntered); //e.g. Fri Nov 13 2015 00:00:00 GMT+0000 (GMT Standard Time)
});
