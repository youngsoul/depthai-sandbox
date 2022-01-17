# Luxonis OAK Cameras Sandbox


## Background Information


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