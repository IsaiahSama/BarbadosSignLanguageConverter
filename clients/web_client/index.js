const video = document.getElementById("videoElement");

const handleStream = (stream) => {
	video.srcObject = stream;
}

const streamError = (err) => {

	console.log("The following error has occurred")
	console.log(err)

	alert(err)

}

if (navigator.mediaDevices.getUserMedia) {
	navigator.mediaDevices.getUserMedia({ video: true })
		.then((stream) => handleStream(stream))
		.catch((err) => streamError(err));
}
