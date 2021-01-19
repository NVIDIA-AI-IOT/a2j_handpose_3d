# Hand Posture Estimation

This repository implements a training pipeline for Anchor to Joint [A2J](https://jonathantompson.github.io/NYU_Hand_Pose_Dataset.htm) paper to train a hand posture estimation model based on depthmaps. We will use NYU Hands dataset to train out model. <br/>
The original authors did a great job publishing their code in the following repository [Link](https://github.com/zhangboshen/A2J) <br/>
The main reason for a new pipeline is to modify number of joints, adding vizualization, and some preprocessing step to be able to run realtime infrence with [**Azure Kinect** camera](https://azure.microsoft.com/en-us/services/kinect-dk/).<br/>
**NOTE** Please refer to the following repository for running realtime inference on a Jetson platform [Link](FILL ME)

* [Download the dataset](#download_dataset)
* [Setup Constants](#setup_constants)
* [Train Model](#train_model)
  * [Visualize Model](#visualize_model)

<a name="download_dataset"></a>
## Download the dataset

Pleae Download the (NYU Hand Pose Dataset)[https://jonathantompson.github.io/NYU_Hand_Pose_Dataset.htm] and unzip the folder.<br/>

<a name="setup_constants"></a>
## Setup Constants

Please modify the following in *pipeline/constants.py*

- DATASET_PATH.
```python
DATASET_PATH="FULL/PATH/TO/DATASET"
```

- (OPTIONAL) SAVE_PATH. this must be the full path to the checkpoint file. by default it will save all the checkpoint files for *PATH/TO/PROJECT/check_point*.
  - Model naming convention: *A2J_BACKBONE_NUMJOINTS_CAMERAPOSE.pth*
  - setup max epoch in pipeline/constants.py*
```python
SAVE_PATH="FULL/PATH/TO/SAVE/DIRECTORY"
```
- (OPTIONAL) BACKBONE_NAME_SWITCHER. This will modify which network is used as the backbone. Set the backbone you want to use to **True**

- (OPTIONAL) CAMERA_VIEW. NYU dataset has been collected with 3 different camera view points. A2J Trained their model on the front view, we will do same. You can also set this value to **ALL** in order to use all the different views.

- (OPTIONAL) NUM_JOINTS. You can set this to either 16 or 36. By default it is set to 16 joints.

<a name="train_model"></a>
## Train Model

Having the dataset ready We need to train the model. <br/>

### 2. train
in *pipeline* dircorty run training by
```bash
python3 train # To start training a model from scratch

python3 train --resume True # To continue an already trained model
python3 train --resume_from_model "FULL/PATH/TO/MODEL" # To continue an already trained model
```

<a name="visualize_model"></a>
### 3. Visualize Model
run training by
```bash
python3 visualize.py # To run inference on a validation image
```
<p align="center">
<img src="readme_files/visulization.png" alt="landing graphic" height="600px"/>
</p>
