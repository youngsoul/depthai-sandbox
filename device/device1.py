import depthai  # depthai - access the camera and its data packets

pipeline = depthai.Pipeline()

# Create nodes, configure them and link them together

# Upload the pipeline to the device
with depthai.Device(pipeline) as device:
  # Print Myriad X Id (MxID), USB speed, and available cameras on the device
  print('MxId:',device.getDeviceInfo().getMxId())
  print('USB speed:',device.getUsbSpeed())
  print('Connected cameras:',device.getConnectedCameras())
