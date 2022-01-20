import numpy as np  # numpy - manipulate the packet data returned by depthai
import cv2  # opencv - display the video stream
import depthai  # depthai - access the camera and its data packets
import blobconverter  # blobconverter - compile and download MyriadX neural network blobs

"""
Online tool to convert models:
https://github.com/luxonis/blobconverter/tree/master/cli


depthai model zoo: https://github.com/luxonis/depthai-model-zoo

https://docs.luxonis.com/projects/api/en/latest/components/nodes/neural_network/

https://blobconverter.luxonis.com
https://blobconverter.luxonis.com/zoo_models?version=2021.4

to see a list of models, on command line:

blobconverter --zoo-list


Here is Github link to NN with size and labels
https://github.com/luxonis/depthai/tree/main/resources/nn


"""

"""
https://docs.luxonis.com/projects/api/en/latest/tutorials/hello_world/

Also used information from this example to pull the labels and label detections
https://docs.luxonis.com/projects/api/en/latest/samples/MobileNet/rgb_mobilenet/#rgb-mobilenetssd

"""

"""
*******
Based On: https://docs.luxonis.com/projects/api/en/latest/tutorials/hello_world/

YoloV3 NN Model
https://github.com/luxonis/depthai/blob/main/resources/nn/tiny-yolo-v3/tiny-yolo-v3.json

Examples
https://docs.luxonis.com/projects/api/en/latest/samples/Yolo/tiny_yolo/
https://docs.luxonis.com/projects/api/en/latest/components/nodes/yolo_detection_network/

"""

tiny_yolo_v3_json = {
    "nn_config":
    {
        "output_format" : "detection",
        "NN_family" : "YOLO",
        "input_size": "416x416",
        "NN_specific_metadata" :
        {
            "classes" : 80,
            "coordinates" : 4,
            "anchors" : [10,14, 23,27, 37,58, 81,82, 135,169, 344,319],
            "anchor_masks" :
            {
                "side26" : [1,2,3],
                "side13" : [3,4,5]
            },
            "iou_threshold" : 0.5,
            "confidence_threshold" : 0.5
        }
    },
    "mappings":
    {
        "labels":
        [
            "person",
            "bicycle",
            "car",
            "motorbike",
            "aeroplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "sofa",
            "pottedplant",
            "bed",
            "diningtable",
            "toilet",
            "tvmonitor",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush"
        ]
    }
}


# color for label text and confidence in BGR order
text_color = (0, 255, 0)


def box_denormalize(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


def main():
    input_size = tiny_yolo_v3_json['nn_config']['input_size'].split('x')
    labelMap = tiny_yolo_v3_json['mappings']['labels']

    pipeline = depthai.Pipeline()

    # Create ColorCamera Node
    cam_rgb = pipeline.create(depthai.node.ColorCamera)
    cam_rgb.setPreviewSize(int(input_size[0]), int(input_size[1]))
    cam_rgb.setInterleaved(False)

    # Create YoloDectionNetwork Node
    detection_nn = pipeline.create(depthai.node.YoloDetectionNetwork)
    # Set path of the blob (NN model). We will use blobconverter to convert&download the model
    # to see the collection of models in the zoo
    # https://github.com/luxonis/depthai/tree/main/resources/nn
    # https://blobconverter.luxonis.com/zoo_models?version=2021.4
    # blobconverter --zoo-list
    detection_nn.setBlobPath(blobconverter.from_zoo(name='yolo-v3-tiny-tf', shaves=6))
    detection_nn.setConfidenceThreshold(0.5)
    # Network specific settings
    detection_nn.setNumClasses(80)
    detection_nn.setCoordinateSize(4)
    detection_nn.setAnchors(np.array([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319]))
    detection_nn.setAnchorMasks({"side26": np.array([1, 2, 3]), "side13": np.array([3, 4, 5])})
    detection_nn.setIouThreshold(0.5)
    detection_nn.setNumInferenceThreads(2)
    detection_nn.input.setBlocking(False)

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
                detections = in_nn.detections

            if frame is not None:
                for detection in detections:
                    # draw the bounding box
                    bbox = box_denormalize(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))

                    # draw the detected bounding box
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

                    # draw the prediction label
                    cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, text_color)

                    # draw the prediction confidence
                    cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, text_color)

                # Show the frame from the OAK device with the detections
                cv2.imshow("Tiny Yolo V3", frame)

            if cv2.waitKey(1) == ord('q'):
                break


if __name__ == "__main__":
    main()
