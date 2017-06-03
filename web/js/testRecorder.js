// https://github.com/bshishov/CognitiveTestPlatform/blob/master/src/web/cognitive_tests/static/js/testRecorder.js

function MyMediaRecorder(options) {
    var self = this;

    this.options = options;

    var recordAudio = options.audio || false;
    var recordVideo = options.video || false;
    var audioRecorder, videoRecorder;
    var localStream;

    var onMediaSuccess = function(stream, callback) {
        localStream = stream;

        var cameraPreviewElement = document.getElementById('cameraPreview');
        if(cameraPreviewElement) {
            cameraPreviewElement.cameraPreview.src = window.URL.createObjectURL(stream);
            cameraPreviewElement.cameraPreview.play();
        }

        if(recordAudio) {
            var options = {
                type: 'audio',
                mimeType: 'audio/wav',
                disableLogs: true,
            }
            audioRecorder = new StereoAudioRecorder(stream, options);
        }

        if(recordVideo) {
            var options = {
                type: 'video',
                //mimeType: 'video/mp4',
                //mimeType: 'video/webm',
                //width: 640,
                //height: 480,
                disableLogs: true,
            }
            //USE WhammyRecorder to record video/mp4 (much slower)
            videoRecorder = new MediaStreamRecorder(stream, options);
        }

        if(recordAudio)
            audioRecorder.record();


        if(recordVideo)
            videoRecorder.record();

        if(callback != undefined) {
            setTimeout(callback, 1000);
        }
    }

    var onMediaError = function(error, callback) {
        console.log(error);
        if(callback != undefined)
            callback(error);
    }

    this.start = function(callback, errorCallback) {
        var constraints = {
            audio: recordAudio,
            video: recordVideo
        };

        if(recordVideo) {
            // Record max allowed resolution
            constraints.video = {
                optional: [
                    {minWidth: 320},
                    {minWidth: 640},
                    {minWidth: 800},
                    {minWidth: 900},
                    {minWidth: 1024},
                    {minWidth: 1280},
                    {minWidth: 1920},
                ]
            };
        }

        console.log("constraints", constraints);

        try {
            navigator.getUserMedia(constraints,
                function(stream) { onMediaSuccess(stream, callback) },
                function(error) { onMediaError(error, errorCallback) }
            );
        } catch (e) {
            alert('MediaRecorder is not supported by this browser.\n\n' +
            'Try Firefox 29 or later, or Chrome 47 or later, with Enable experimental Web Platform features enabled from chrome://flags.');
            console.error('Exception while creating MediaRecorder:', e);
            return;
        }
    }

    this.stop = function(callback) {
        var remaining = 0;
        var wait = function(){
            if(--remaining <= 0 && callback != undefined) {
                localStream.stop();
                callback();
            }
        };

        if(recordAudio) {
            remaining++;
            audioRecorder.stop(wait);
        }

        if(recordVideo) {
            remaining++;
            videoRecorder.stop(wait);
        }
    }

    // Only after stop
    this.getVideoBlob = function() {
        //return videoRTC.getBlob();
        return videoRecorder.blob;
    }

    this.getAudioBlob = function() {
        //return audioRTC.getBlob();
        return audioRecorder.blob;
    }
}

function TestRecorder(element, options) {
    this.isRunning = false;
    this.recordAudio = options.audio;
    this.recordVideo = options.video;
    this.recordMedia = options.video || options.audio;
    this.post_to = options.post_to || "";
    this.additional_data = options.additional_data;

    this.mediaRecorder = new MyMediaRecorder({
        audio: options.audio,
        video: options.video
    });

    var events = [];
    var startTime = -1;
    var self = this;

    this.start = function(callback) {
        this.startTime = Date.now();

        if(this.recordMedia)
            this.mediaRecorder.start(callback, function(err) {
                alert(err.name);
            });
        else if(callback != undefined)
            callback();

        this.isRunning = true;
    }

    this.stop = function(callback) {
        if(this.recordMedia)
        {
            this.mediaRecorder.stop(function() {
                if(callback != undefined)
                    callback();
            });
        }
        else if(callback != undefined)
                callback();

        self.isRunning = false;
    }

    function saveFile (name, data, type="octet/stream") {
        var blob = data instanceof Blob ? data : new Blob([data], { type: type });
        if (data != null && navigator.msSaveBlob)
            return navigator.msSaveBlob(blob, name);
        var url = window.URL.createObjectURL(blob);

        var a = document.createElement("a");
        document.body.appendChild(a);
        a.style = "display: none";

        a.href = url;
        a.download = name;
        a.click();
        window.URL.revokeObjectURL(url);
    }

    this.send = function(callback) {
        if(this.post_to) {
            var formData = new FormData();

            if(this.additional_data) {
                for (var key in this.additional_data){
                    if (this.additional_data.hasOwnProperty(key)) {
                        formData.append(key, this.additional_data[key]);
                    }
                }
            }

            formData.append('events', JSON.stringify(events, null, 4));

            if(this.recordAudio)
                formData.append('audio', this.mediaRecorder.getAudioBlob(), "audio.wav");

            if(this.recordVideo)
                formData.append('video', this.mediaRecorder.getVideoBlob(), "video.webm");

            var xhr = new XMLHttpRequest();
            xhr.open("POST", this.post_to, true);
            xhr.send(formData);
            xhr.onload = callback;
        }
        else {
            saveFile("events.json", JSON.stringify(events, null, 4));
            if(this.recordAudio)
                saveFile("audio.wav", this.mediaRecorder.getAudioBlob());
            if(this.recordVideo)
                saveFile("video.webm", this.mediaRecorder.getVideoBlob());
        }
    }

    this.logEvent = function(name, args) {
        if(this.isRunning) {
            events.push({
                time: this.getTime(),
                name: name,
                args: args
            });
        }
    }

    this.getTime = function() {
        return Date.now() - this.startTime;
    }
}

function TestManager(recorder) {
    var element = document.createElement('div');
    var eventStart = new Event('start');
    var eventRecordingStop = new Event('recordingStop');
    var eventComplete = new Event('complete');
    var eventSendComplete = new Event('sendComplete');
    var eventSendFail = new Event('sendFail');
    var self = this;

    this.recorder = recorder;

    this.log = function(eventName, args) {
        this.recorder.logEvent(eventName, args);
        if(eventName.indexOf("mouse") < 0)
            console.log("Test event:", this.getTime(), eventName, args);
    }

    this.start = function() {
        recorder.start(function() {
            self.log("test_start");
            element.dispatchEvent(eventStart);
        });
    }

    this.complete = function() {
        self.log("test_complete");
        recorder.stop(function() {
            element.dispatchEvent(eventRecordingStop); // RECORDING STOPPED
            recorder.send(function() {
                element.dispatchEvent(eventSendComplete); // SEND COMPLETE
            }, function() {
                element.dispatchEvent(eventSendFail); // SEND FAIL
            });
        });
        element.dispatchEvent(eventComplete); // COMPLETE CALLED
    }

    this.on = function(eventName, callback) {
        element.addEventListener(eventName, callback);
    }

    this.getTime = function() {
        return this.recorder.getTime();
    }
}