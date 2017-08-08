# Mental Workload estimation with RNN
This is the set of tools and approaches to estimate mental workload on video sequence. To measure the performance of the human-machine system, the human performance must be taken into account. However, it remains difficult to build a well-generalized model of the human condition. Thus we need to focus on modeling of the separate factors. Mental workload estimation is the crucial task in system design and its performance analysis. There are several existing approaches for estimating mental workload, but most of them are not yet appropriate enough for practical use. In this study, the method of estimating mental workload is proposed. The idea is to estimate workload from the video sequence from a common camera (i.e. webcam) using recurrent neural networks (RNN) trained individually. First, the workload is estimated under special experimental conditions using task-based approaches while facial, and eye movement features are extracted from the video captured during the experiment. Then, extracted workload is modeled based on the extracted training data with RNN with long short-term memory (LSTM). As a result, low-error LSTM models are trained.

The work was presented at 2017 ASRTU China-Russia International Conference on Intelligent Manufacturing at Harbin, China. [Full-text article available here](https://github.com/bshishov/MentalWorkload/blob/master/article.pdf).

# Installation
The software consists of two parts: The first one is the web-based (located in `/web`) test for recording training data for separate individual which will be used for further model training. And the other part is the model itself (located in `/model`) which is built with python, [keras](https://keras.io/) over [tensorflow](https://www.tensorflow.org/) and [tools for processing facial video sequences](https://github.com/TadasBaltrusaitis/OpenFace).

## Requirements for the model
* Python 2.7 or 3
* Tensorflow for Python (refer to installation guide [here](https://www.tensorflow.org/install/))
* [OpenFace toolkit](https://github.com/TadasBaltrusaitis/OpenFace) (you will only need already built executables)
* Python packages:
  * **Numpy** for handling the data
  * **Pandas** for initial data processing (`pip install pandas`)
  * **Keras** for deep-learning stuff (`pip install keras`)
  * **Matplotlib** for plotting results (`pip install matplotlib`)
 
Most of these packages (except Keras and TF) are already packaged in [Anaconda](https://www.continuum.io/downloads). If you have any problems installing the requirements you could simply contact me. 

# Usage
## Web-based test
First, make sure your webcamera is attached and properly positioned. In order to run the web test and capture video you will need to run the `/web/index.html` file. It could be opened locally so just open it in your browser, allow usage of your webcamera and hit **Start** button. After clicking some circles the script will automatically *download* captured data:
* `video.webm`: a recorded video fragment
* `events.json`: recorded mouse and tests events. Which contains informtaion about you reaction time etc.

You will need these files for processing them into the training data for our model.

## Test results processing
First, we need to extract features (Action Units) from the video. To do that we will use the open source [OpenFace toolkit](https://github.com/TadasBaltrusaitis/OpenFace) and its feature extraction module.
Then we will need to estimate mental workload from our test performance (circle click reaction times and so on).

### Extracting features from video
Find  `/model/openface.py` file and chacne the `OPENFACE_FEATURE_EXTRACTION_PATH` according to the path to the OpenFace executables.
Then you can run `python openface.py <PATH-TO-YOUR-VIDEO-WEBM-FILE>`. This will produce a `csv` file containg feature values for each frame of the video.

## Training the model
Simply run `python model/main.py <video_webm_path> <events_json_path>`. This will run the training procedure for a given files. 
The trained Keras model will be saved in `<video_file_name>.h5` (HDF5 format) file.

## Evaluating the model
Once the model is trained you can use it to evaluate on any data. Please refer for [model loading in Keras](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model).
