import cv2  # opencv - display the video stream
import depthai  # depthai - access the camera and its data packets

def main():
    # Create empty processing pipeline
    pipeline = depthai.Pipeline()

    # cam_rgb is a ColorCamera node.
    # https://docs.luxonis.com/projects/api/en/latest/components/nodes/color_camera/
    cam_rgb = pipeline.create(depthai.node.ColorCamera)
    cam_rgb.setPreviewSize(400, 400)
    # Set planar or interleaved data of preview output frames
    cam_rgb.setInterleaved(False)

    xout_rgb = pipeline.create(depthai.node.XLinkOut)
    xout_rgb.setStreamName('rgb')
    cam_rgb.preview.link(xout_rgb.input)

    with depthai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue("rgb")
        frame = None

        while True:
            in_rgb = q_rgb.tryGet()
            if in_rgb is not None:
                frame = in_rgb.getCvFrame()
                if frame is not None:
                    cv2.imshow("OAK-1", frame)
            if cv2.waitKey(1) == ord('q'):
                break

if __name__ == '__main__':
    main()