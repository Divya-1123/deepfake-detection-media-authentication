OpenCV DNN face detector (SSD, ResNet-10 backbone), the model shipped with OpenCV's dnn samples.
- deploy.prototxt: https://github.com/opencv/opencv/blob/4.x/samples/dnn/face_detector/deploy.prototxt
- res10_300x300_ssd_iter_140000.caffemodel: https://github.com/opencv/opencv_3rdparty/tree/dnn_samples_face_detector_20170830
If these files are missing the app falls back to the Haar cascade bundled with OpenCV.
