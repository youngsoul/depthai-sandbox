import numpy as np  # numpy - manipulate the packet data returned by depthai
import cv2  # opencv - display the video stream
import depthai  # depthai - access the camera and its data packets
import blobconverter  # blobconverter - compile and download MyriadX neural network blobs

"""
https://docs.luxonis.com/projects/api/en/latest/tutorials/hello_world/
"""
def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

def main():
    pipeline = depthai.Pipeline()

    # Create ColorCamera Node
    cam_rgb = pipeline.create(depthai.node.ColorCamera)
    cam_rgb.setPreviewSize(300,300)
    cam_rgb.setInterleaved(False)

    # Create MobileNetDetectionNetwork Node
    detection_nn = pipeline.create(depthai.node.MobileNetDetectionNetwork)
    # Set path of the blob (NN model). We will use blobconverter to convert&download the model
    # detection_nn.setBlobPath("/path/to/model.blob")
    detection_nn.setBlobPath(blobconverter.from_zoo(name='mobilenet-ssd', shaves=6))
    detection_nn.setConfidenceThreshold(0.5)

    # connect the ColorCamera output to the MobileNet input
    cam_rgb.preview.link(detection_nn.input)

    # Create XLinkOut nodes to send data to host for the camera
    xout_rgb = pipeline.create(depthai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    # connect ColorCamera to XLinkOut
    cam_rgb.preview.link(xout_rgb.input)

    # Create XLinkOut nodes to send data to host for the neural network
    xout_nn = pipeline.create(depthai.node.XLinkOut)
    xout_nn.setStreamName("nn")

    # connect neural network to XLinkOut
    detection_nn.out.link(xout_nn.input)

    # get the virtual device, and loop forever reading messages
    # from the internal queue
    with depthai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue("rgb")
        q_nn = device.getOutputQueue("nn")

        frame = None
        detections = []

        while True:
            in_rgb = q_rgb.tryGet()
            in_nn = q_nn.tryGet()

            if in_rgb is not None:
                frame = in_rgb.getCvFrame()

            if in_nn is not None:
                detections = in_nn.detections

            if frame is not None:
                for detection in detections:
                    bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                cv2.imshow("preview", frame)

            if cv2.waitKey(1) == ord('q'):
                break

if __name__ == "__main__":
    main()
