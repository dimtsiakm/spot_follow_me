#!/usr/bin/env python3
"""Capture one color frame from each of Spot's body cameras, rotate each one
upright using the robot's own calibration, optionally detect people, then
save to disk and display."""

import os
from math import atan2, degrees

import cv2
import numpy as np

import bosdyn.client
import bosdyn.client.util
from bosdyn.api import image_pb2
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, VISION_FRAME_NAME, get_a_tform_b
from bosdyn.client.image import ImageClient, build_image_request

ROBOT_IP = '192.168.80.3'
OUTPUT_DIR = 'captures'
JPEG_QUALITY = 90

RUN_DETECTION = True
MODEL_PATH = 'yolo11n.pt'
CONFIDENCE = 0.35
PERSON_CLASS_ID = 0          # 'person' in COCO

SOURCES = [
    'frontleft_fisheye_image',
    'frontright_fisheye_image',
    'left_fisheye_image',
    'right_fisheye_image',
    'back_fisheye_image',
]


def upright_angle(image_response, base_frame=VISION_FRAME_NAME):
    """Exact counter-clockwise degrees needed to make 'up' in base_frame
    point up in the image. Read from this robot's calibration, not a constant."""
    shot = image_response.shot
    tform = get_a_tform_b(shot.transforms_snapshot, base_frame,
                          shot.frame_name_image_sensor)
    if tform is None:
        raise RuntimeError(f'no transform {base_frame} -> {shot.frame_name_image_sensor}')

    R = tform.rotation.to_matrix()          # columns = camera axes in base_frame
    up = np.array([0.0, 0.0, 1.0])
    u_x = float(np.dot(up, R[:, 0]))        # "up" projected onto image-right
    u_y = float(np.dot(up, R[:, 1]))        # "up" projected onto image-down
    return degrees(atan2(u_y, u_x)) + 90.0


def rotate_bound(img, angle_deg):
    """Rotate about the centre without cropping the corners."""
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    M[0, 2] += nw / 2.0 - cx
    M[1, 2] += nh / 2.0 - cy
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def decode(image_response):
    """Return a BGR numpy array regardless of how the robot encoded the frame."""
    shot = image_response.shot
    buf = np.frombuffer(shot.image.data, dtype=np.uint8)

    if shot.image.format == image_pb2.Image.FORMAT_JPEG:
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    pixel_format = shot.image.pixel_format
    if pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
        frame = buf.reshape(shot.image.rows, shot.image.cols, 3)
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if pixel_format == image_pb2.Image.PIXEL_FORMAT_RGBA_U8:
        frame = buf.reshape(shot.image.rows, shot.image.cols, 4)
        return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    frame = buf.reshape(shot.image.rows, shot.image.cols)
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


def load_model():
    from ultralytics import YOLO
    model = YOLO(MODEL_PATH)
    model.fuse()
    return model


def detect_people(model, frame):
    """Return a list of (x1, y1, x2, y2, confidence) in pixel coordinates."""
    results = model.predict(frame, classes=[PERSON_CLASS_ID],
                            conf=CONFIDENCE, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    return [(int(a), int(b), int(c), int(d), float(s))
            for (a, b, c, d), s in zip(xyxy, conf)]


def draw(frame, detections):
    for x1, y1, x2, y2, score in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'person {score:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = load_model() if RUN_DETECTION else None

    sdk = bosdyn.client.create_standard_sdk('SpotColorFrames')
    robot = sdk.create_robot(ROBOT_IP)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    image_client = robot.ensure_client(ImageClient.default_service_name)

    available = {s.name for s in image_client.list_image_sources()}
    sources = [s for s in SOURCES if s in available]
    if not sources:
        raise RuntimeError('none of the configured sources exist on this robot')

    requests = [
        build_image_request(name,
                            pixel_format=image_pb2.Image.PIXEL_FORMAT_RGB_U8,
                            quality_percent=JPEG_QUALITY)
        for name in sources
    ]
    responses = image_client.get_image(requests)

    total = 0
    print(f'{"source":28s} {"size":>11s} {"mount":>9s} {"applied":>9s} {"people":>7s}')
    for response in responses:
        name = response.source.name
        frame = decode(response)

        mount = upright_angle(response, BODY_FRAME_NAME)
        applied = upright_angle(response, VISION_FRAME_NAME)
        upright = rotate_bound(frame, applied)

        detections = detect_people(model, upright) if model else []
        total += len(detections)
        if detections:
            upright = draw(upright, detections)

        path = os.path.join(OUTPUT_DIR, f'{name}.jpg')
        cv2.imwrite(path, upright)

        size = f'{frame.shape[1]}x{frame.shape[0]}'
        count = str(len(detections)) if model else '-'
        print(f'{name:28s} {size:>11s} {mount:+9.3f} {applied:+9.3f} {count:>7s}')

        cv2.imshow(name, upright)

    if model:
        print(f'\n{total} person detection(s) across {len(responses)} frames')
    print(f'saved to {os.path.abspath(OUTPUT_DIR)} — press any key in a window to close')
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()