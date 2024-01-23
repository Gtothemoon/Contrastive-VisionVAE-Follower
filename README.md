# Contrastive-VisionVAE-Follower
Contrastive-VisionVAE-Follower is a model used for multi-modal task called Vision-and-Language Navigation (VLN). The Contrastive-VisionVAE-Follower model was proposed in my undergraduate graduation thesis. It is an improvement on the Speaker-Follower model<sup>[1].

<div align=center>
  <img src="https://github.com/Gtothemoon/Contrastive-VisionVAE-Follower/blob/main/contrastive-VisionVAE-follower.png" alt="Contrastive-VisionVAE-Follower">
</div>
<p align="center">Figure 1: Illustration of contrast-visionVAE-follower.</p>

## Abstract
A robot that can clearly understand human language and conduct intelligent navigation in the real visual world can perform specific tasks for human beings, such as bridge inspection, fire fighting and so on. Researchers hope that the robot can understand natural language instructions, and perform corresponding actions to reach the designated destination in the real environment combined with visual information. This navigation task is called Visual-and-Language Navigation.

Aiming at the room to room task in Visual-and-Language Navigation, based on the follower model, this paper studies and improves it into the contrast-visionVAE-follower model. By adding the cross modal contrastive learning module to learn the matching relationship between language and visual cross modal information, and by adding the visual variational auto-encoder module, the visual information is encoded and reconstructed under the condition of limited training data, so as to increase the diversity of data during training and improve the generalization performance of the model in an unprecedented visual environment. The experimental results show that the navigation success rate of contrast-visionVAE-follower model on the validation data set is higher than that of follower model, and the navigation error is smaller, which has better navigation performance.

## Vision-and-Language Navigation (VLN)
The idea that we might be able to give general, verbal instructions to a robot and have at least a reasonable probability that it will carry out the required task is one of the long-held goals of robotics, and artificial intelligence (AI). Despite significant progress, there are a number of major technical challenges that need to be overcome before robots will be able to perform general tasks in the real world. One of the primary requirements will be new techniques for linking natural language to vision and action in unstructured, previously unseen environments. It is the navigation version of this challenge that we refer to as Vision-and-Language Navigation (VLN)<sup>[2].

<div align=center>
  <img src="https://github.com/Gtothemoon/Contrastive-VisionVAE-Follower/blob/main/R2R.png" alt="R2R">
</div>
<p align="center">Figure 2: Room-to-Room (R2R) navigation task. We focus on executing natural language navigation instructions in previously unseen real-world buildings. The agent’s camera can be rotated freely. Blue discs indicate nearby (discretized) navigation options<sup>[2].</p>

## References
[1] [Speaker-Follower Models for Vision-and-Language Navigation](https://proceedings.neurips.cc/paper/2018/hash/6a81681a7af700c6385d36577ebec359-Abstract.html) (NeurIPS 2018)

[2] [Vision-and-Language Navigation: Interpreting visually-grounded navigation instructions in real environments](https://openaccess.thecvf.com/content_cvpr_2018/html/Anderson_Vision-and-Language_Navigation_Interpreting_CVPR_2018_paper.html) (CVPR 2018)

## Instructions
### Download the R2R dataset and Matterport3DSimulator files

Download the [R2R dataset](https://bringmeaspoon.org/) required for the experiment. First, enter the official website of the R2R dataset, sign a usage agreement, and then send the signed usage agreement to the official designated email address matterport3d@googlegroups.com. You will soon receive an official reply, which will include a Python file for downloading the R2R dataset. Execute the Python file on the server command line to download the R2R dataset. The complete R2R dataset size is approximately 1.3T. As this experiment only requires the simulation function of the Matterport3DSimulator, only the `matterport_skybox_images` and `undistorted_camera_parameters` data from the R2R dataset need to be downloaded. You should unzip and save the downloaded data in this folder: `contrastive-VisionVAE-follower/data/share/patternport3d/mP3Ddata`

Then download the complete file of the [Matterport3DSimulator](https://github.com/peteanderson80/Matterport3DSimulator). Use the git command to copy the Matterport3DSimulator file from the github website to the corresponding folder on the local server. Afterwards, it is necessary to set environment variables and associate the Matterport3DSimulator with the R2R dataset.

### Install Matterport3DSimulator dependencies
It is recommended to use Dockerfile to install the Matterport3DSimulator, so we need to install docker first (please refer to the tutorial of [Matterport3DSimulator](https://github.com/peteanderson80/Matterport3DSimulator)). Docker is an open-source software containerized application engine that allows developers to package their applications and dependency files into a portable image container when building their various applications. Then, the corresponding image container can be easily released to any platform, such as machines using Linux or Windows operating systems. Of course, the Matterport3DSimulator can also be installed without using Dockerfile, but installing without Docker can be cumbersome and require users to carefully consider the relationships between the required dependent files, making installation more difficult. This experiment uses Dockerfile to install the Matterport3DSimulator.

Before using docker to install Matterport3DSimulator, the configuration on the server needs to meet the following requirements:

* Nvidia GPU with driver >= 396.37;
* Install [docker](https://docs.docker.com/engine/install/);
* Install [nvidia-docker2.0](https://github.com/NVIDIA/nvidia-docker).

### Install Matterport3DSimulator
【❗When the installation of Matterport3DSimulator using Docker fails due to network packet loss, the Dockerfile in that folder: `/Dockerfiles` can be used to replace the original Dockerfile. The Dockerfile in this folder have been swapped sources (using NetEase source).】

Use the Dockerfile to install the Matterport3DSimulator. Enter the folder of the previously downloaded Matterport3DSimulator. There is a Dockerfile in the folder for building the docker image corresponding to the simulator. Enter the following instructions in the command line of the server:
```
docker build -t mattersim:9.2-devel-ubuntu18.04 .
```

The docker image of Matterport3DSimulator will be automatically pulled from the remote to the local server, and the installation will be completed after a period of time. Then run the docker container and mount the Matterport3DSimulator file and R2R dataset:
```
nvidia-docker run -it --mount type=bind,source=$MATTERPORT_DATA_DIR,target=/root/mount/Matterport3DSimulator/data/v1/scans --volume `pwd`:/root/mount/Matterport3DSimulator mattersim:9.2-devel-ubuntu18.04
```

After successfully entering the docker container, keep your command line path still at the location of the Matterport3DSimulator folder, and enter the following instructions in the command line:
```
cd /root/mount/Matterport3DSimulator
mkdir build && cd build
cmake -DEGL_RENDERING=ON ..
make
cd ../
```

After waiting for a period of time, the Matterport3DSimulator and environment have been set up. In order to speed up data loading and reduce memory usage during model training, the images under the `matterport_skybox_images` file in the R2R dataset need to be preprocessed by reducing the size and merging all cube faces into one image. Run the following instructions in the docker container:
```
./scripts/downsize_skybox.py
```

Once completed, the `matterport_skybox_images` subdirectory in the dataset will contain image files with filenames in the format <PANO_ID>_skybox_small.jpg.

### Training demo
Start the installed Matterport3DSimulator on the server, enter the corresponding file directory `contrastive-VisionVAE-follower/tasks/R2R` where the code corresponding to the contrast-visionVAE-follower model is stored, and enter the following instructions in the server command line to start the model training process:
```
python3 tasks/R2R/train.py
```

### Evaluation metrics
* Navigation error (NE): Defined as the straight-line path distance between the agent’s last stopping position and the target position.
* Success rate (SR): In a specific navigation task, if the straight-line shortest path distance between the agent's final stopping position and the target position is less than the threshold of 3m, the task is considered successful. When multiple navigation tasks are performed, the ratio between the number of successes in all navigation tasks and the total number of navigation tasks is the success rate.
