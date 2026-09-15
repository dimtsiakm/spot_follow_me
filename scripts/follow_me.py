#!/usr/bin/env python3
"""Follow-me demo: Spot tracks a person across its 5 body cameras and walks
to keep FOLLOW_DISTANCE between them. If nobody is in view, it stands still
with its body pitched up ("looking up"), searching, until someone appears.

A small Tkinter window shows how many distinct people are currently visible
and who is locked, with Left/Right buttons to switch the lock to the next
person to that side.

Distance/heading to a person come from the depth cameras (no arm, no
external tracker hardware) - each *_depth_in_visual_frame image is
pixel-aligned with its *_fisheye_image counterpart, so a detection's pixel
centroid can be looked up directly in the matching depth image, back-projected
to a 3D point with that source's pinhole intrinsics, and transformed into the
body frame with the robot's own calibration (get_a_tform_b), the same
approach grab_frame.py uses for uprighting images.

Usage:
    python scripts/follow_me.py
"""

import json
import logging
import os
import time
import tkinter as tk
from datetime import datetime
from math import atan2, degrees, hypot

import cv2
import numpy as np

import bosdyn.client
import bosdyn.client.util
from bosdyn.geometry import EulerZXY
from bosdyn.api import image_pb2, robot_state_pb2
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, VISION_FRAME_NAME, get_a_tform_b
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.power import safe_power_off
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         blocking_sit, blocking_stand)
from bosdyn.client.robot_state import RobotStateClient

ROBOT_IP = '192.168.80.3'

MODEL_PATH = 'yolo11n.pt'
CONFIDENCE = 0.50
PERSON_CLASS_ID = 0          # 'person' in COCO
TRACKER_CFG = 'bytetrack.yaml'

CAMERAS = ['frontleft', 'frontright', 'left', 'right', 'back']
JPEG_QUALITY = 85

FOLLOW_DISTANCE = 1.5        # m, the gap Spot tries to hold to the person
DISTANCE_DEADBAND = 0.2      # m, no forward/back motion inside this band
MIN_SAFE_DISTANCE = 0.8      # m, hard floor: never drive closer than this
OBSTACLE_AVOIDANCE_PADDING = 0.5   # m, extra clearance Spot's own obstacle
                                    # avoidance keeps from *real* obstacles
                                    # (walls, furniture) on top of its
                                    # footprint - unset, this defaults to 0.0
                                    # extra padding, so the robot's built-in
                                    # avoidance can silently override/refuse
                                    # a velocity command much closer than
                                    # that; MIN_SAFE_DISTANCE above is a
                                    # separate, unrelated floor for distance
                                    # to the *followed person*.
MAX_LINEAR_VEL = 0.8         # m/s
MAX_ANGULAR_VEL = 1.0        # rad/s
KP_LINEAR = 0.6
KP_ANGULAR = 1.2

MIN_CMD_DURATION = 0.5       # s, floor/first-guess for how long a velocity command stays valid
MAX_CMD_DURATION = 2.0       # s, ceiling - caps how long a stalled loop can keep it moving
TARGET_LOST_FRAMES = 8       # loop iterations a locked target may vanish for
IDLE_REFRESH_SEC = 2.0       # re-issue the idle pose this often while idle
FULL_SCAN_INTERVAL_SEC = 1.0 # once locked, only re-poll all 5 cameras this often

# Body pitch, radians. footprint_R_body rotates body -> footprint(gravity)
# frame; positive pitch tips the nose down (bows), so a negative value tips
# the nose/cameras up. See bosdyn.geometry.EulerZXY.to_quaternion.
IDLE_LOOK_UP_PITCH = -0.4

SHOW_WINDOWS = True
LOOP_DELAY_MS = 1            # gap between iterations, just enough for Tk to process clicks

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')

DEPTH_PATCH = 5               # side length (px) of the median-depth sample window

# Two raw detections (from any cameras, e.g. overlapping frontleft/frontright)
# whose body-frame positions are closer than this are treated as one physical
# person, so the people count doesn't double up in camera overlap zones.
DEDUP_RADIUS = 0.6           # m
# A locked person is followed frame-to-frame by nearest body-frame position,
# not by (camera, track_id) - that lets the lock survive a handoff between
# cameras. A jump larger than this is treated as "lost", not "moved".
MAX_LOCK_JUMP = 1.0          # m


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def decode_color(image_response):
    shot = image_response.shot
    buf = np.frombuffer(shot.image.data, dtype=np.uint8)
    if shot.image.format == image_pb2.Image.FORMAT_JPEG:
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    frame = buf.reshape(shot.image.rows, shot.image.cols, 3)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def decode_depth(image_response):
    shot = image_response.shot
    buf = np.frombuffer(shot.image.data, dtype=np.uint16)
    return buf.reshape(shot.image.rows, shot.image.cols)


def upright_angle(image_response, base_frame=VISION_FRAME_NAME):
    """Exact counter-clockwise degrees needed to make 'up' in base_frame
    point up in the image. Read from this robot's calibration, not a
    constant - same approach as grab_frame.py."""
    shot = image_response.shot
    tform = get_a_tform_b(shot.transforms_snapshot, base_frame,
                          shot.frame_name_image_sensor)
    if tform is None:
        raise RuntimeError(f'no transform {base_frame} -> {shot.frame_name_image_sensor}')

    R = tform.rotation.to_matrix()
    up = np.array([0.0, 0.0, 1.0])
    u_x = float(np.dot(up, R[:, 0]))
    u_y = float(np.dot(up, R[:, 1]))
    return degrees(atan2(u_y, u_x)) + 90.0


def rotation_matrix(h, w, angle_deg):
    """Affine matrix that rotates an h x w image upright without cropping
    corners, plus the resulting canvas size. YOLO is trained on upright
    photos and misses people reliably in Spot's sideways-mounted body
    cameras, so detection has to run on the rotated frame - but depth and
    the pinhole intrinsics are defined on the RAW frame, so this matrix also
    lets detections be mapped back for the depth lookup."""
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    M[0, 2] += nw / 2.0 - cx
    M[1, 2] += nh / 2.0 - cy
    return M, nw, nh


def rotate_bound(img, M, nw, nh):
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def unrotate_point(M, x, y, raw_w, raw_h):
    """Map a pixel from the rotated/upright frame back to raw sensor
    coordinates, for sampling depth and applying the raw-frame intrinsics."""
    M_inv = cv2.invertAffineTransform(M)
    raw_x = M_inv[0, 0] * x + M_inv[0, 1] * y + M_inv[0, 2]
    raw_y = M_inv[1, 0] * x + M_inv[1, 1] * y + M_inv[1, 2]
    raw_x = clamp(raw_x, 0, raw_w - 1)
    raw_y = clamp(raw_y, 0, raw_h - 1)
    return int(round(raw_x)), int(round(raw_y))


def sample_depth_m(depth_raw, depth_scale, cx, cy, patch=DEPTH_PATCH):
    """Median of the nonzero depth readings in a small window, in meters."""
    half = patch // 2
    y0, y1 = max(0, cy - half), min(depth_raw.shape[0], cy + half + 1)
    x0, x1 = max(0, cx - half), min(depth_raw.shape[1], cx + half + 1)
    window = depth_raw[y0:y1, x0:x1]
    valid = window[window > 0]
    if valid.size == 0:
        return None
    return float(np.median(valid)) / depth_scale


def person_in_body_frame(image_response, cx, cy, depth_m):
    """Back-project a color-image pixel + depth reading to a 3D point in
    BODY_FRAME_NAME, using this source's own pinhole intrinsics and this
    shot's own calibration snapshot."""
    intrinsics = image_response.source.pinhole.intrinsics
    fx, fy = intrinsics.focal_length.x, intrinsics.focal_length.y
    px, py = intrinsics.principal_point.x, intrinsics.principal_point.y

    x = (cx - px) * depth_m / fx
    y = (cy - py) * depth_m / fy
    z = depth_m

    shot = image_response.shot
    tform = get_a_tform_b(shot.transforms_snapshot, BODY_FRAME_NAME,
                          shot.frame_name_image_sensor)
    if tform is None:
        return None
    return tform.transform_point(x, y, z)


def load_model():
    from ultralytics import YOLO
    model = YOLO(MODEL_PATH)
    model.fuse()
    return model


def track_people(model, frame, conf):
    """Run the persistent tracker on one camera's frame. Returns a list of
    (track_id, x1, y1, x2, y2, confidence). track_id is stable across calls
    for *this* model instance, so each camera needs its own model/tracker -
    interleaving cameras through one tracker would corrupt its motion model."""
    results = model.track(frame, persist=True, tracker=TRACKER_CFG,
                          classes=[PERSON_CLASS_ID], conf=conf, verbose=False)
    boxes = results[0].boxes
    if boxes is None or boxes.id is None:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    ids = boxes.id.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    return [(int(i), int(a), int(b), int(c), int(d), float(s))
            for i, (a, b, c, d), s in zip(ids, xyxy, conf)]


def dedupe_people(raw_detections):
    """Greedily cluster raw per-camera detections by body-frame proximity so
    the same physical person seen by two cameras (e.g. the overlapping
    frontleft/frontright pair) is counted once. Returns clusters sorted by
    bearing, left-to-right is not assumed - ascending bearing means rightmost
    (most negative) first, since body-frame +y is left of the robot."""
    clusters = []
    for det in raw_detections:
        for cluster in clusters:
            cx = sum(m['x'] for m in cluster) / len(cluster)
            cy = sum(m['y'] for m in cluster) / len(cluster)
            if hypot(det['x'] - cx, det['y'] - cy) < DEDUP_RADIUS:
                cluster.append(det)
                break
        else:
            clusters.append([det])

    people = []
    for cluster in clusters:
        best = max(cluster, key=lambda m: m['conf'])
        avg_x = sum(m['x'] for m in cluster) / len(cluster)
        avg_y = sum(m['y'] for m in cluster) / len(cluster)
        people.append({
            'camera': best['camera'], 'track_id': best['track_id'],
            'x': avg_x, 'y': avg_y,
            'distance': hypot(avg_x, avg_y), 'bearing': atan2(avg_y, avg_x),
            'members': cluster,
        })
    people.sort(key=lambda p: p['bearing'])
    return people


def build_requests(camera_names):
    requests = []
    for name in camera_names:
        requests.append(build_image_request(f'{name}_fisheye_image',
                                            pixel_format=image_pb2.Image.PIXEL_FORMAT_RGB_U8,
                                            quality_percent=JPEG_QUALITY))
        requests.append(build_image_request(f'{name}_depth_in_visual_frame'))
    return requests


def mobility_params_with_padding(**kwargs):
    params = RobotCommandBuilder.mobility_params(**kwargs)
    params.obstacle_params.obstacle_avoidance_padding = OBSTACLE_AVOIDANCE_PADDING
    return params


def send_velocity(command_client, vx, vy, vrot, duration):
    params = mobility_params_with_padding()
    cmd = RobotCommandBuilder.synchro_velocity_command(vx, vy, vrot, params=params,
                                                        frame_name='body')
    command_client.robot_command(cmd, end_time_secs=time.time() + duration)


def send_idle_pose(command_client):
    footprint_R_body = EulerZXY(yaw=0.0, roll=0.0, pitch=IDLE_LOOK_UP_PITCH)
    params = mobility_params_with_padding(footprint_R_body=footprint_R_body)
    cmd = RobotCommandBuilder.synchro_stand_command(params=params)
    command_client.robot_command(cmd)


def send_stop(command_client):
    command_client.robot_command(RobotCommandBuilder.stop_command())


def setup_run_logger():
    """One JSON-lines file per calendar day (logs/log_YYYY-MM-DD.jsonl),
    appended across runs, so state/classification history carries over
    between sessions instead of living only in the terminal scrollback."""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f'log_{datetime.now():%Y-%m-%d}.jsonl')
    logger = logging.getLogger('follow_me.run')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.FileHandler(log_path)
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
    return logger


def log_event(logger, event, **fields):
    record = {'ts': datetime.now().isoformat(timespec='milliseconds'), 'event': event}
    record.update(fields)
    logger.info(json.dumps(record))


class FollowState:
    def __init__(self):
        self.lock_pos = None     # (x, y) in body frame of the locked person
        self.locked_camera = None
        self.lost_count = 0
        self.people = []         # current frame's deduped people, sorted by bearing
        self.is_idle = True
        self.last_idle_refresh = 0.0
        self.running = True
        self.mode = 'stand'      # 'sit' or 'stand'
        self.follow_enabled = False
        self.last_loop_dt = MIN_CMD_DURATION
        self.last_full_scan = 0.0


def nearest_index(people, pos):
    return min(range(len(people)),
              key=lambda i: hypot(people[i]['x'] - pos[0], people[i]['y'] - pos[1]))


def main():
    logger = setup_run_logger()
    log_event(logger, 'session_start', follow_distance=FOLLOW_DISTANCE,
              distance_deadband=DISTANCE_DEADBAND, min_safe_distance=MIN_SAFE_DISTANCE,
              target_lost_frames=TARGET_LOST_FRAMES, confidence=CONFIDENCE)

    models = {name: load_model() for name in CAMERAS}

    sdk = bosdyn.client.create_standard_sdk('SpotFollowMe')
    robot = sdk.create_robot(ROBOT_IP)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    image_client = robot.ensure_client(ImageClient.default_service_name)
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    state_client = robot.ensure_client(RobotStateClient.default_service_name)

    available = {s.name for s in image_client.list_image_sources()}
    cameras = [name for name in CAMERAS
              if f'{name}_fisheye_image' in available
              and f'{name}_depth_in_visual_frame' in available]
    if not cameras:
        raise RuntimeError('none of the configured cameras (with depth) exist on this robot')
    full_requests = build_requests(cameras)
    single_requests = {name: build_requests([name]) for name in cameras}

    state = FollowState()

    root = tk.Tk()
    root.title('Follow Me')
    label_var = tk.StringVar(value='Starting...')
    tk.Label(root, textvariable=label_var, font=('Helvetica', 12), justify='left',
            anchor='w').pack(padx=12, pady=(12, 6), fill='x')

    confidence_var = tk.DoubleVar(value=CONFIDENCE)
    conf_frame = tk.Frame(root)
    conf_frame.pack(padx=12, pady=(0, 6), fill='x')
    tk.Label(conf_frame, text='Detection confidence').pack(anchor='w')
    tk.Scale(conf_frame, from_=0.1, to=0.9, resolution=0.05, orient='horizontal',
            variable=confidence_var).pack(fill='x')

    def reset_to_idle_pose():
        state.is_idle = True
        send_idle_pose(command_client)
        state.last_idle_refresh = time.time()

    def update_follow_button():
        if state.follow_enabled:
            follow_btn.config(text='Follow-Me: ON', bg='#2ecc71', activebackground='#27ae60')
        else:
            follow_btn.config(text='Follow-Me: OFF', bg='#95a5a6', activebackground='#7f8c8d')

    def on_sit():
        state.follow_enabled = False
        update_follow_button()
        send_stop(command_client)
        state.mode = 'sit'
        blocking_sit(command_client, timeout_sec=10)
        log_event(logger, 'mode_change', mode='sit', via='button')

    def on_stand():
        state.mode = 'stand'
        blocking_stand(command_client, timeout_sec=10)
        reset_to_idle_pose()
        log_event(logger, 'mode_change', mode='stand', via='button')

    def on_toggle_follow():
        state.follow_enabled = not state.follow_enabled
        if state.follow_enabled and state.mode == 'sit':
            state.mode = 'stand'
            blocking_stand(command_client, timeout_sec=10)
        if not state.follow_enabled and state.mode == 'stand':
            send_stop(command_client)
            reset_to_idle_pose()
        update_follow_button()
        log_event(logger, 'follow_toggle', follow_enabled=state.follow_enabled, mode=state.mode)

    def on_stop():
        state.follow_enabled = False
        update_follow_button()
        send_stop(command_client)
        if state.mode == 'stand':
            reset_to_idle_pose()
        log_event(logger, 'stop_button')

    mode_row = tk.Frame(root)
    mode_row.pack(pady=(0, 6))
    tk.Button(mode_row, text='Sit', width=10, command=on_sit).pack(side='left', padx=6)
    tk.Button(mode_row, text='Stand', width=10, command=on_stand).pack(side='left', padx=6)
    follow_btn = tk.Button(mode_row, text='Follow-Me: OFF', width=14, command=on_toggle_follow)
    follow_btn.pack(side='left', padx=6)
    update_follow_button()

    def cycle_lock(direction):
        people = state.people
        if not people:
            return
        if state.lock_pos is None:
            idx = 0 if direction > 0 else len(people) - 1
        else:
            idx = (nearest_index(people, state.lock_pos) + direction) % len(people)
        chosen = people[idx]
        state.lock_pos = (chosen['x'], chosen['y'])
        state.locked_camera = chosen['camera']
        state.lost_count = 0

    arrow_row = tk.Frame(root)
    arrow_row.pack(pady=(0, 12))
    tk.Button(arrow_row, text='◀ Left', width=10,
             command=lambda: cycle_lock(-1)).pack(side='left', padx=6)
    tk.Button(arrow_row, text='Right ▶', width=10,
             command=lambda: cycle_lock(1)).pack(side='left', padx=6)

    def shutdown():
        if not state.running:
            return
        state.running = False
        robot.logger.info('Stopping and powering off')
        log_event(logger, 'session_end')
        try:
            send_stop(command_client)
        finally:
            safe_power_off(command_client, state_client)
        if SHOW_WINDOWS:
            cv2.destroyAllWindows()
        root.destroy()

    stop_canvas = tk.Canvas(root, width=90, height=90, highlightthickness=0)
    stop_canvas.pack(pady=(0, 14))
    circle = stop_canvas.create_oval(5, 5, 85, 85, fill='#e74c3c', outline='#922b21', width=3)
    stop_text = stop_canvas.create_text(45, 45, text='STOP', fill='white',
                                        font=('Helvetica', 13, 'bold'))
    stop_canvas.tag_bind(circle, '<Button-1>', lambda e: on_stop())
    stop_canvas.tag_bind(stop_text, '<Button-1>', lambda e: on_stop())

    root.protocol('WM_DELETE_WINDOW', shutdown)

    def step():
        if not state.running:
            return
        if state.mode == 'sit':
            label_var.set('Sitting')
            root.after(200, step)
            return
        loop_start = time.time()
        try:
            # Fetching+decoding+tracking all 5 cameras every loop is the
            # actual cost driver (5 color + 5 depth images over the network,
            # not the ~5ms/frame local YOLO inference). Once locked, poll
            # only the camera that's currently seeing the person, and only
            # do a full 5-camera sweep periodically to catch a handoff to a
            # different camera or refresh the "people detected" count.
            need_full_scan = (state.lock_pos is None
                              or loop_start - state.last_full_scan > FULL_SCAN_INTERVAL_SEC
                              or state.locked_camera not in cameras)
            if need_full_scan:
                active_cameras = cameras
                active_requests = full_requests
                state.last_full_scan = loop_start
            else:
                active_cameras = [state.locked_camera]
                active_requests = single_requests[state.locked_camera]

            responses = image_client.get_image(active_requests)
            by_camera = {r.source.name.replace('_fisheye_image', ''): r
                        for r in responses if r.source.name.endswith('_fisheye_image')}
            depth_by_camera = {r.source.name.replace('_depth_in_visual_frame', ''): r
                               for r in responses
                               if r.source.name.endswith('_depth_in_visual_frame')}

            raw = []            # camera, track_id, bbox, conf, x, y (body frame)
            frames = {}         # camera -> upright color frame, for the preview windows
            for name in active_cameras:
                color_resp = by_camera[name]
                depth_resp = depth_by_camera[name]
                raw_frame = decode_color(color_resp)
                raw_h, raw_w = raw_frame.shape[:2]
                angle = upright_angle(color_resp, VISION_FRAME_NAME)
                M, nw, nh = rotation_matrix(raw_h, raw_w, angle)
                upright_frame = rotate_bound(raw_frame, M, nw, nh)
                frames[name] = upright_frame
                depth_raw = decode_depth(depth_resp)
                depth_scale = depth_resp.source.depth_scale

                # YOLO is fed the upright frame (it misses people reliably on
                # Spot's sideways-mounted body cameras), but depth and the
                # pinhole intrinsics are defined on the raw sensor grid, so
                # the tracked centroid is mapped back before looking either up.
                for track_id, x1, y1, x2, y2, conf in track_people(
                        models[name], upright_frame, confidence_var.get()):
                    cx_up, cy_up = (x1 + x2) // 2, (y1 + y2) // 2
                    cx_raw, cy_raw = unrotate_point(M, cx_up, cy_up, raw_w, raw_h)
                    depth_m = sample_depth_m(depth_raw, depth_scale, cx_raw, cy_raw)
                    if depth_m is None:
                        continue
                    point = person_in_body_frame(color_resp, cx_raw, cy_raw, depth_m)
                    if point is None:
                        continue
                    bx, by_, _ = point
                    raw.append({'camera': name, 'track_id': track_id,
                               'bbox': (x1, y1, x2, y2), 'conf': conf,
                               'x': bx, 'y': by_})

            frame_people = dedupe_people(raw)
            if need_full_scan:
                # Only the full sweep sees every camera, so only it can give
                # an accurate total - a locked-camera-only poll would make
                # the count (and Left/Right cycling) flicker down to ~1.
                state.people = frame_people
            people = state.people

            target = None
            if state.lock_pos is not None and frame_people:
                idx = nearest_index(frame_people, state.lock_pos)
                if hypot(frame_people[idx]['x'] - state.lock_pos[0],
                        frame_people[idx]['y'] - state.lock_pos[1]) < MAX_LOCK_JUMP:
                    target = frame_people[idx]
                    state.lock_pos = (target['x'], target['y'])
                    state.locked_camera = target['camera']
                    state.lost_count = 0
            if target is None and state.lock_pos is not None:
                state.lost_count += 1
                if state.lost_count > TARGET_LOST_FRAMES:
                    log_event(logger, 'lock_lost', last_lock_pos=state.lock_pos,
                              last_camera=state.locked_camera, lost_count=state.lost_count)
                    state.lock_pos = None
                    state.locked_camera = None

            if target is None and state.lock_pos is None and frame_people:
                idx = min(range(len(frame_people)), key=lambda i: frame_people[i]['distance'])
                target = frame_people[idx]
                state.lock_pos = (target['x'], target['y'])
                state.locked_camera = target['camera']
                state.lost_count = 0
                log_event(logger, 'lock_acquired', track_id=target['track_id'],
                          camera=target['camera'], distance=target['distance'])

            # A missed detection on a single frame (classifier miss, brief
            # occlusion) must not drop the robot out of follow motion - that
            # was flickering moving -> idle-look-up-stand -> moving every time
            # a frame came back empty. The lock (state.lock_pos) already
            # survives up to TARGET_LOST_FRAMES misses; drive motion off that
            # lock, falling back to the last known position/bearing when this
            # particular frame has no detection to match, and only fall back
            # to the idle pose once the lock itself is actually cleared above.
            if state.follow_enabled and state.lock_pos is not None:
                if target is not None:
                    distance, bearing = target['distance'], target['bearing']
                    track_desc = f'track {target["track_id"]} via {target["camera"]}'
                else:
                    distance = hypot(*state.lock_pos)
                    bearing = atan2(state.lock_pos[1], state.lock_pos[0])
                    track_desc = (f'last known position (miss {state.lost_count}/'
                                  f'{TARGET_LOST_FRAMES})')
                error = distance - FOLLOW_DISTANCE
                vx = 0.0 if abs(error) < DISTANCE_DEADBAND else clamp(
                    KP_LINEAR * error, -MAX_LINEAR_VEL, MAX_LINEAR_VEL)
                if distance < MIN_SAFE_DISTANCE:
                    vx = min(vx, 0.0)
                vrot = clamp(KP_ANGULAR * bearing, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)
                cmd_duration = clamp(state.last_loop_dt * 2.0, MIN_CMD_DURATION, MAX_CMD_DURATION)
                send_velocity(command_client, vx, 0.0, vrot, cmd_duration)
                state.is_idle = False
                lock_text = (f'Locked: {track_desc}\n'
                            f'distance {distance:.2f} m   bearing {bearing:+.2f} rad')
            else:
                if not state.is_idle:
                    send_idle_pose(command_client)
                    state.last_idle_refresh = time.time()
                    state.is_idle = True
                    log_event(logger, 'idle_enter', follow_enabled=state.follow_enabled)
                elif time.time() - state.last_idle_refresh > IDLE_REFRESH_SEC:
                    send_idle_pose(command_client)
                    state.last_idle_refresh = time.time()
                if not state.follow_enabled:
                    lock_text = (f'Follow-Me OFF - selected track {target["track_id"]} via '
                                f'{target["camera"]} (not moving)' if target is not None
                                else 'Follow-Me OFF - standing by')
                else:
                    lock_text = 'Locked: none - searching' if not people else 'Locked: none'

            label_var.set(f'People detected: {len(people)}\n{lock_text}')

            log_event(logger, 'tick', mode=state.mode, follow_enabled=state.follow_enabled,
                      is_idle=state.is_idle, people_count=len(people),
                      lock_pos=state.lock_pos, locked_camera=state.locked_camera,
                      lost_count=state.lost_count,
                      target=({'track_id': target['track_id'], 'camera': target['camera'],
                               'distance': round(target['distance'], 3),
                               'bearing': round(target['bearing'], 3)}
                              if target is not None else None),
                      detections=[{'camera': d['camera'], 'track_id': d['track_id'],
                                   'conf': round(d['conf'], 3),
                                   'distance': round(hypot(d['x'], d['y']), 3)} for d in raw])

            if SHOW_WINDOWS:
                by_cam_raw = {}
                for d in raw:
                    by_cam_raw.setdefault(d['camera'], []).append(d)
                target_members = target['members'] if target is not None else []
                for name, frame in frames.items():
                    for d in by_cam_raw.get(name, []):
                        x1, y1, x2, y2 = d['bbox']
                        is_target = any(m is d for m in target_members)
                        color = (0, 255, 0) if is_target else (128, 128, 128)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f'id{d["track_id"]} {hypot(d["x"], d["y"]):.1f}m',
                                   (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                                   0.5, color, 1, cv2.LINE_AA)
                    cv2.imshow(name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    shutdown()
                    return

            state.last_loop_dt = time.time() - loop_start
        except Exception as exc:
            robot.logger.error(f'follow loop failed: {exc}')
            log_event(logger, 'error', message=str(exc))
            shutdown()
            return

        if state.running:
            root.after(LOOP_DELAY_MS, step)

    with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        # Start from whatever the robot is already doing - don't force a
        # stand if it was sitting when the script was launched.
        initial_behavior = state_client.get_robot_state().behavior_state.state
        already_standing = initial_behavior in (
            robot_state_pb2.BehaviorState.STATE_STANDING,
            robot_state_pb2.BehaviorState.STATE_STEPPING,
        )

        robot.logger.info('Powering on')
        robot.power_on(timeout_sec=20)

        if already_standing:
            state.mode = 'stand'
            blocking_stand(command_client, timeout_sec=10)
            send_idle_pose(command_client)
            state.last_idle_refresh = time.time()
        else:
            state.mode = 'sit'

        root.after(LOOP_DELAY_MS, step)
        try:
            root.mainloop()
        except KeyboardInterrupt:
            pass
        finally:
            shutdown()


if __name__ == '__main__':
    main()
