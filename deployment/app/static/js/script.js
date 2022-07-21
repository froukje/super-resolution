var loadFile = function(event) {
    var image = document.getElementById('input');
	image.src = URL.createObjectURL(event.target.files[0]);
    };

var showImg = function(event) {
var upload = document.getElementById('uploaded-img');
var upscaled = document.getElementById('upscaled-img');
    upload.classList.toggle("show");
    upscaled.classList.toggle("not-show");
}

var showLoader = function(event) {
    document.getElementById("wrapper").style.display = "block";
    document.getElementById("hero").style.opacity = 0.5;
  }