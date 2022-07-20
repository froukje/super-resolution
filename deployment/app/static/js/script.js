var loadFile = function(event) {
    var image = document.getElementById('input');
	image.src = URL.createObjectURL(event.target.files[0]);
    };

var showImg = function(event) {
var upload = document.getElementById('uploaded-img');
    upload.classList.toggle("show");
}

var showUpImg = function(event){
var upscaled = document.getElementById('img-up');
    upscaled.classList.toggle("show");
    //event.preventDefault();
}