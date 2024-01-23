# Vision-and-Language Navigation (VLN)
The idea that we might be able to give general, verbal instructions to a robot and have at least a reasonable probability that it will carry out the required task is one of the long-held goals of robotics, and artificial intelligence (AI). Despite significant progress, there are a number of major technical challenges that need to be overcome before robots will be able to perform general tasks in the real world. One of the primary requirements will be new techniques for linking natural language to vision and action in unstructured, previously unseen environments. It is the navigation version of this challenge that we refer to as Vision-and-Language Navigation (VLN)<sup>[1].

# Contrastive-VisionVAE-Follower
Contrastive-VisionVAE-Follower is a model used for multi-modal task called Vision-and-Language Navigation (VLN). The Contrastive-VisionVAE-Follower model was proposed in my undergraduate graduation thesis. It is an improvement on the Speaker-Follower model<sup>[2].

# References
[1] [Vision-and-Language Navigation: Interpreting visually-grounded navigation instructions in real environments](https://openaccess.thecvf.com/content_cvpr_2018/html/Anderson_Vision-and-Language_Navigation_Interpreting_CVPR_2018_paper.html) (CVPR 2018)

[2] [Speaker-Follower Models for Vision-and-Language Navigation](https://proceedings.neurips.cc/paper/2018/hash/6a81681a7af700c6385d36577ebec359-Abstract.html) (NeurIPS 2018)

# Instructions
**Download the R2R dataset and Matterport3DSimulator files**

Download the [R2R dataset](https://bringmeaspoon.org/) required for the experiment. First, enter the official website of the R2R dataset, sign a usage agreement, and then send the signed usage agreement to the official designated email address matterport3d@googlegroups.com. You will soon receive an official reply, which will include a Python file for downloading the R2R dataset. Execute the Python file on the server command line to download the R2R dataset. The complete R2R dataset size is approximately 1.3T. As this experiment only requires the simulation function of the Matterport3DSimulator, only the "matterport_skybox_images" and "undistorted_camera_parameters" data from the R2R dataset need to be downloaded. You should unzip and save the downloaded data in this folder: contrastive-VisionVAE-follower/data/share/patternport3d/mP3Ddata

**Download the R2R dataset and Matterport3DSimulator files**
