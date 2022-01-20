# Luxonis OAK Cameras Sandbox


## Background Information

### YouTube Videos

[Hello World](https://youtu.be/iUNQPspeUA8)

[Video Feed](https://youtu.be/DbgX7gQa304)

### Medium Posts

[Post1](https://patrick-ryan.medium.com/walk-through-of-opencv-ai-kit-using-the-mobilenet-detection-algorithm-fdbfe9531e54)

[Post 2](https://patrick-ryan.medium.com/the-opencv-ai-kit-oak-super-simple-video-streamer-starter-fff707630f16)

### Pipelines

Pipeline first steps

To get DepthAI up and running, you have to create a pipeline, populate it with nodes, configure the nodes and link them together. After that, the pipeline can be loaded onto the Device and be started.


## Install Libraries

```shell
pip install numpy opencv-python depthai blobconverter
```


## Reference Material

[Luxonis API Documentation](https://docs.luxonis.com/projects/api/en/latest/)

[Luxonis SDK Documentation](https://docs.luxonis.com/projects/sdk/en/latest/)

## RaspberryPI

If you have setup OpenCV on the RaspberryPI then you should only need to install the following into your cv2 environment:

Use [this script](https://github.com/youngsoul/rpi_opencv_install/blob/master/pi_install_imagelibs.sh) to install OpenCV along with image libraries. 

```shell
pip install depthai blobconverter
```

[Luxonis RaspberryPI Setup](https://docs.luxonis.com/projects/api/en/latest/install/#raspberry-pi-os)

If you get an error like:

```text
RuntimeError: Failed to find device (ma2480), error message: X_LINK_DEVICE_NOT_FOUND
```

See this [troubleshooting guide](https://docs.luxonis.com/en/latest/pages/troubleshooting/)

* Unplug OAK Device

* Run the following:
```shell
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

* Plug the OAK device back into USB3 slot

If your system still does not work, run the folllowing:

```shell
sudo curl -fL https://docs.luxonis.com/install_dependencies.sh | bash
```
From [Luxonis RaspberryPI Setup](https://docs.luxonis.com/projects/api/en/latest/install/#raspberry-pi-os)

## Examples

### Simple Video Feed

Inside of the `simple-camera-feed` directory, there is a script called, `videostream.py`.  

This was the basis for the Gitrepo below, and the [Medium Post](https://patrick-ryan.medium.com/the-opencv-ai-kit-oak-super-simple-video-streamer-starter-fff707630f16)

[Source Code]https://github.com/youngsoul/depthai-video-stream

