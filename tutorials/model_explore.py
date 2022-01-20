import numpy as np  # numpy - manipulate the packet data returned by depthai
import cv2  # opencv - display the video stream
import depthai  # depthai - access the camera and its data packets
import blobconverter  # blobconverter - compile and download MyriadX neural network blobs
import argparse


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
Because of the answer to this question:
https://discuss.luxonis.com/d/228-determine-nn-input-size-automatically

I am creating an explorer app

Here is Github link to NN with size and labels
https://github.com/luxonis/depthai/tree/main/resources/nn


YOLO
https://docs.luxonis.com/projects/api/en/latest/components/nodes/yolo_detection_network/
https://docs.luxonis.com/projects/api/en/latest/samples/Yolo/tiny_yolo/

"""

models = {
    "mobilenet": {
        "labelMap":["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
            "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
            "sheep", "sofa", "train", "tvmonitor"],
        "inputSize": (300,300),
        "modelName": "mobilenet-ssd",
        "node": depthai.node.MobileNetDetectionNetwork

    },
    "tinyyolo": {
        "labelMap":[
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
        ],
        "inputSize":(416,416),
        "modelName": "tiny-yolo-v3",
        "node": depthai.node.YoloDetectionNetwork
    }
}


# MobilenetSSD label texts
labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
            "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
            "sheep", "sofa", "train", "tvmonitor"]

# color for label text and confidence in BGR order
text_color = (255, 0, 0)


def box_denormalize(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

def __cli__():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model',
        help='Model to use',
        default='mobilenet',
        type=str,
        choices=["mobilenet", "tinyyolo"],
    )
    parser.add_argument('--confidence', required=False, default=0.5, type=float, help="Confidence level 0-1")

    args = vars(parser.parse_args())

    model_name = args['model']
    confidence = args['confidence']

    return model_name, confidence


def main():
    model_name, confidence = __cli__()
    model = models[model_name]
    print(f"Using Model: {model['modelName']}")
    print(f"Input Image Size: {model['inputSize']}")

    pipeline = depthai.Pipeline()

    # Create ColorCamera Node
    cam_rgb = pipeline.create(depthai.node.ColorCamera)
    cam_rgb.setPreviewSize(model['inputSize'][0], model['inputSize'][1])
    # cam_rgb.setPreviewSize(416,416)
    cam_rgb.setInterleaved(False)

    # Create MobileNetDetectionNetwork Node
    detection_nn = pipeline.create(model['node'])
    # Set path of the blob (NN model). We will use blobconverter to convert&download the model
    # to see the collection of models in the zoo
    # https://github.com/luxonis/depthai/tree/main/resources/nn
    # tiny_yolo_path = '/Users/patrickryan/.cache/blobconverter/tiny-yolo-v3_openvino_2021.4_6shave.blob'
    # image size is 416x416
    # detection_nn.setBlobPath(tiny_yolo_path)
    detection_nn.setBlobPath(blobconverter.from_zoo(name=model['modelName'], shaves=6))
    detection_nn.setConfidenceThreshold(confidence)
    if model['modelName'] == 'tinyyolo':
        detection_nn.setNumClasses(80)
        detection_nn.setCoordinateSize(4)
        detection_nn.setAnchors(np.array([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319]))
        detection_nn.setAnchorMasks({"side26": np.array([1, 2, 3]), "side13": np.array([3, 4, 5])})
        detection_nn.setIouThreshold(0.5)

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
                    if detection.label < len(model['labelMap']):
                        cv2.putText(frame, model['labelMap'][detection.label], (bbox[0] + 10, bbox[1] + 20),
                                    cv2.FONT_HERSHEY_TRIPLEX, 0.5, text_color)
                    else:
                        cv2.putText(frame, f"Unknown label index: {detection.label}", (bbox[0] + 10, bbox[1] + 20),
                                    cv2.FONT_HERSHEY_TRIPLEX, 0.5, text_color)

                    # draw the prediction confidence
                    if int(detection.confidence*100) > 100:
                        print(detection.confidence)

                    cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40),
                                cv2.FONT_HERSHEY_TRIPLEX, 0.5, text_color)

                # Show the frame from the OAK device with the detections
                cv2.imshow("MobileNetDetections", frame)

            if cv2.waitKey(1) == ord('q'):
                break


if __name__ == "__main__":
    main()
