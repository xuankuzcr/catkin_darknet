# catkin_darknet_RM

## Brief Introduction

This is a ROS package developed for enemy robot detection in ICRA-Robomaster AI Challenge. In the following ROS package you are able to use YOLO (V3) on **GPU and CPU**. </br>

The pre-trained model of the convolutional neural network is able to detect pre-trained classes including `the data set made from the front, back, left and right sides of the robot and the numbers on the armor plate`.</br>
 At the same time, the real time robot pose can be calculated by PnP, which is shown in the terminal.

The  packages have been tested under ROS Kinetic and Ubuntu 16.04. The effect is shown as follows.

<img src="img/Yolo_tiny recognition.gif" alt="show" />

## How to build

The compilation method is the same as the normal ROS function package.

## How to run

```
    roslaunch darknet_ros darknet_ros.launch
    roslaunch usb_cam usb_cam.launch
```

## Notice

The interface including the RGB image input topic and the robot 6DOF  pose output topic have been reserved.
