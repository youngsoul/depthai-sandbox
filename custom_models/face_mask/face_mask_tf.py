import numpy as np  # numpy - manipulate the packet data returned by depthai
import cv2  # opencv - display the video stream
import depthai  # depthai - access the camera and its data packets
import blobconverter  # blobconverter - compile and download MyriadX neural network blobs

# color for label text and confidence in BGR order
text_color = (255, 0, 0)


def box_denormalize(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


def main():
    pipeline = depthai.Pipeline()

    # Create ColorCamera Node
    cam_rgb = pipeline.create(depthai.node.ColorCamera)
    cam_rgb.setPreviewSize(224, 224)
    # cam_rgb.setPreviewSize(416,416)
    cam_rgb.setInterleaved(False)

    # Create MobileNetDetectionNetwork Node
    detection_nn = pipeline.create(depthai.node.NeuralNetwork)
    # Set path of the blob (NN model). We will use blobconverter to convert&download the model
    # to see the collection of models in the zoo
    # https://github.com/luxonis/depthai/tree/main/resources/nn
    # tiny_yolo_path = '/Users/patrickryan/.cache/blobconverter/tiny-yolo-v3_openvino_2021.4_6shave.blob'
    # image size is 416x416
    detection_nn.setBlobPath("/Users/patrickryan/.cache/blobconverter/saved_model_openvino_2021.4_6shave.blob")

    # connect the ColorCamera preview output to the MobileNet input
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
                # NN can output from multiple layers. Print all layer names:
                print("------------------")
                # print(in_nn.getAllLayerNames())
                [print(f"Layer name: {l.name}, Type: {l.dataType}, Dimensions: {l.dims}") for l in in_nn.getAllLayers()]
                mask, no_mask = in_nn.getLayerFp16('StatefulPartitionedCall/model/dense_1/Softmax')
                print(f"Mask[{round(mask,1)}], No Mask[{round(no_mask,1)}]")

            if frame is not None:
                # Show the frame from the OAK device
                cv2.imshow("TF Face Mask", frame)

            if cv2.waitKey(1) == ord('q'):
                break


if __name__ == "__main__":
    main()
