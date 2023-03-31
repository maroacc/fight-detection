# Surveillance Camera Fight Detection
This project aims to detect fights in surveillance camera videos using a CNN + LSTM model trained on the Fight Detection Surveillance Dataset. We used a pretrained CNN InceptionV3 as the backbone for our model and optimized the hyperparameters of the LSTM.

## Description
Surveillance cameras are a common feature in public places such as malls, public transport, and night clubs. However, monitoring these cameras in real-time is a daunting task, especially in large facilities with numerous cameras. This is where our project comes in. By automatically detecting fights in surveillance camera footage, our system can alert security personnel to violent incidents as soon as they occur.

The model consists of a pretrained InceptionV3 CNN, followed by a 2-layer LSTM. The layers of the CNN were frozen due to the small dataset, and the hyperparameters of the LSTM were optimized. Various models with data augmentation were also tried to improve performance.

## Installation
To install the necessary dependencies for this project, run the following command:
# Fight Detection Video Classification


## Tennis image classification using InceptionV3:
Tennis image classification using a pretrained InceptionV3 base model.
This is a guide on how to execute it in Google Colab

Tennis video classification using a pretrained InceptionV3 base model + a LSTM architecture.
This is a guide on how to execute it in Google Colab

1. Download the THETIS RGB dataset from <http://thetis.image.ece.ntua.gr/>
2. Upload the THETIS zipfile to Google Drive
3. Execute the preprocessing_thetis_non_seq.ipynb notebook
4. Execute the training_thetis_non_seq.ipynb notebook

## Naive model:
Tennis video classification using a pretrained InceptionV3 base model + calculating the average for each video.
This is a guide on how to execute it in Google Colab

Tennis video classification using a pretrained InceptionV3 base model + a LSTM architecture.
This is a guide on how to execute it in Google Colab

1. Download the THETIS RGB dataset from <http://thetis.image.ece.ntua.gr/>
2. Upload the THETIS zipfile to Google Drive
3. Execute the preprocessing_thetis_non_seq.ipynb notebook
4. Execute the training_thetis_non_seq.ipynb to obtain the weights of the image classificator.
5. Execute the naive_model.ipynb notebook

## Tennis video classification using InceptionV3 + LSTM:

Tennis video classification using a pretrained InceptionV3 base model + a LSTM architecture.
This is a guide on how to execute it in Google Colab

1. Download the THETIS RGB dataset from <http://thetis.image.ece.ntua.gr/>
2. Upload the THETIS zipfile to Google Drive
3. Open the InceptionV3-LSTM.ipynb file is Google Collab
4. Unzip the dataset
5. Place the videos from the dataset in content/data/train and content/data/test folders. Each video type should have its own folder

>	| data/train
> >		| Forehand
> >		| Backhand
> >		...
>	| data/test
> >		| Forehand
> >		| Backhand
> >		...

6. Clone the Github repository into Google Colab
7. Extract files from video with script extract_files.py. Pass video files extenssion as a param

`	$ python extract_files.py mp4`

8. Check the data_file.csv and choose the acceptable sequence length of frames. It should be less or equal to lowest one if you want to process all videos in dataset. We recommend a length of 43 frames (avg -2*std).
9. Extract sequence for each video with InceptionV3 and train LSTM. Run train.py script with sequence_length, class_limit, image_height, image_width args

`	$ python predict.py 43 2 480 640`

10. Save your best model file. (For example, lstm-features.hdf5)
11. Evaluate your model using predict.py. It will generate an .xlsx with the confusion matrix and the predictions for each video.

`	$ python train.py 75 2 720 1280`

12. Use clasify.py script to clasify your video. Args sequence_length, class_limit, saved_model_file, video_filename

`	$ python clasify.py 75 2 lstm-features.hdf5 video_file.mp4`

The result will be placed in result.avi file.

# Requirements

Ignore if you are using Google Colab

This code requires you have Keras 2 and TensorFlow 1 or greater installed. Please see the `requirements.txt` file. To ensure you're up to date, run:

`pip install -r requirements.txt`

You must also have `ffmpeg` installed in order to extract the video files.

# Saved models

The weights of the models trained by us is too big to upload to Github. If you wish to use it contact us
