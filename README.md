<p align="center">
  <img src="assets/santanna_logo.png" alt="Scuola Superiore Sant'Anna, Pisa" width="400">
</p>

# Spot: follow-me

A person-following demo for the Boston Dynamics **Spot** robot. It uses only
Spot's own body cameras, with no arm, no extra sensors and no external tracker.

Spot detects and tracks people across all five of its body cameras. It then
uses the depth cameras to locate each person in 3D and walks to stay a set
distance (1.5 m by default) from the person it is locked onto. A small control
window lets the operator see how many people are in view, switch the lock to
another person, sit or stand the robot, and stop it at any time.

**Video demo:** [watch on YouTube](https://www.youtube.com/watch?v=pd_SZVHmwis)

**Author:** Dimitrios Tsiakmakis, MSc AI, PhD Candidate, Scuola Superiore Sant'Anna, Pisa
([website](https://dimtsiakm.github.io/) · [LinkedIn](https://www.linkedin.com/in/dimitris-tsiakmakis/))

---

## Contents

| Script | What it does |
| --- | --- |
| [`scripts/follow_me.py`](scripts/follow_me.py) | The main follow-me demo (GUI and camera previews). |
| [`scripts/grab_frame.py`](scripts/grab_frame.py) | Takes one picture from each body camera, rotates it upright, runs person detection and saves the results to `captures/`. Use it to check that the cameras and the model work before running the demo. |
| [`scripts/basic_commands.py`](scripts/basic_commands.py) | A minimal terminal menu that powers on the robot and makes it sit or stand. Use it to check that you can connect, authenticate and take the lease. |

## How it works

1. **Image capture.** Each loop requests a color image (`*_fisheye_image`) and
   a matching depth image (`*_depth_in_visual_frame`) from the cameras
   `frontleft`, `frontright`, `left`, `right` and `back`. Once a person is
   locked, the script polls only the camera that sees them and runs a full
   five-camera scan once per second, which keeps network load low.
2. **Uprighting.** Spot's body cameras are mounted sideways, and YOLO misses
   people in rotated images. Each frame is therefore rotated upright using the
   rotation angle read from the robot's own calibration, not a fixed value.
3. **Detection and tracking.** [YOLO11n](https://docs.ultralytics.com/) with
   ByteTrack runs on each upright frame. Each camera has its own tracker so
   that track IDs stay stable.
4. **3D localisation.** The centre of each detection box is mapped back to
   raw sensor pixels. The script reads the median depth around that pixel,
   back-projects it with the camera's pinhole intrinsics, and transforms it
   into the robot's body and vision frames. Detections more than 55° off a
   camera's optical axis are discarded, because the pinhole model does not
   hold near the edge of the fisheye lenses.
5. **De-duplication and locking.** Detections from different cameras that
   are within 0.6 m of each other count as one person. The lock is stored in
   the vision (world-fixed) frame, so it survives the robot turning and the
   person moving from one camera's view to another's.
6. **Control.** A proportional controller sets forward and turning speeds.
   It smooths the bearing, ignores small errors (deadbands), limits how fast
   the turning speed can change, and holds back forward speed until the robot
   faces the person. It never drives closer than 0.8 m, and Spot's own
   obstacle avoidance keeps an extra 0.5 m of padding.

Every session writes a JSON-lines log to `logs/log_YYYY-MM-DD.jsonl` (one
file per day), recording state changes, lock events and per-tick detections.

---

## Requirements

### Hardware

- A **Boston Dynamics Spot** with the standard five body cameras (color and
  depth), running Spot software 4.x or later.
- A **laptop** connected to the robot's network (by default Spot's own Wi-Fi
  access point, robot IP `192.168.80.3`).
- An **NVIDIA GPU** in the laptop is strongly recommended. Inference runs on
  the laptop, and the demo was developed on an RTX 4050 Laptop GPU. It also
  runs on CPU, but the control loop will be slower.
- An **E-Stop** for the robot, such as the Spot tablet or controller, or the
  SDK's `estop_gui` example. These scripts do not register an E-Stop endpoint
  themselves.

### Software

- **Python 3.10** (recommended; this is the version the demo was developed
  with). Make sure the Spot SDK release you install supports your Python
  version.
- Python packages, listed in [`requirements.txt`](requirements.txt):
  - `bosdyn-client`, `bosdyn-api`, `bosdyn-core` ≥ 4.0
  - `ultralytics` ≥ 8.3 (YOLO11) and `lap` (needed by ByteTrack)
  - `opencv-python`, `numpy`
- **PyTorch with CUDA** for GPU inference (see step 3 below).
- **Tkinter** for the control window. It comes with the Python installers
  for Windows and macOS. On Debian/Ubuntu, install it with
  `sudo apt install python3-tk`.
- **Spot user credentials** (username and password for the robot).

---

## Installation

The commands below are for Windows (PowerShell). The equivalent Linux/macOS
commands are shown in comments.

### 1. Clone the repository

```bash
git clone <repository-url> spot-follow-me
cd spot-follow-me
```

### 2. Create a virtual environment

The project's `.gitignore` expects the virtual environment to be named `spot-env`.

```powershell
py -3.10 -m venv spot-env
.\spot-env\Scripts\Activate.ps1
# Linux/macOS:  python3.10 -m venv spot-env && source spot-env/bin/activate

python -m pip install --upgrade pip
```

### 3. Install PyTorch with CUDA (GPU users)

Install PyTorch **before** the other packages, so that `ultralytics` does not
install the CPU-only build. Pick the command that matches your CUDA version on
the [PyTorch "Get Started" page](https://pytorch.org/get-started/locally/).
For example:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Skip this step if you only want to run on CPU.

### 4. Install the remaining dependencies

```powershell
pip install -r requirements.txt
```

### 5. Check the GPU (optional)

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

This should print `True` followed by the name of your GPU.

### 6. Model weights

The scripts load `yolo11n.pt` from the **current working directory**.
Ultralytics downloads it automatically on first run if it is not there. The
file is git-ignored, so it is not part of the repository.

---

## Configuration

Settings are constants at the top of each script. Edit them there.

| Constant | Where | Default | Meaning |
| --- | --- | --- | --- |
| `ROBOT_IP` | all scripts | `192.168.80.3` | Robot address. The default is Spot's address on its own Wi-Fi access point; change it if you connect through a different network. |
| `FOLLOW_DISTANCE` | `follow_me.py` | `1.5` m | Distance Spot tries to keep from the person. |
| `MIN_SAFE_DISTANCE` | `follow_me.py` | `0.8` m | Spot never drives closer than this to the person. |
| `MAX_LINEAR_VEL` / `MAX_ANGULAR_VEL` | `follow_me.py` | `0.8` m/s / `1.0` rad/s | Speed limits. |
| `CONFIDENCE` | `follow_me.py` | `0.50` | Initial detection threshold (also adjustable live in the GUI). |
| `SHOW_WINDOWS` | `follow_me.py` | `True` | Show the camera preview windows. |
| `RUN_DETECTION` | `grab_frame.py` | `True` | Run person detection on the captured frames. |

### Credentials

The scripts log in with `bosdyn.client.util.authenticate`. You can set your
credentials as environment variables so you aren't asked for them on every
run:

```powershell
$env:BOSDYN_CLIENT_USERNAME = "user"
$env:BOSDYN_CLIENT_PASSWORD = "password"
# Linux/macOS:  export BOSDYN_CLIENT_USERNAME=user BOSDYN_CLIENT_PASSWORD=password
```

If these variables are not set, the script asks for the username and password
in the terminal.

---

## Usage

Run all scripts **from the repository root**, with the virtual environment
active. Run them from there because `yolo11n.pt` and `captures/` are looked up
relative to the folder you run the command from.

Before running:

1. Connect the laptop to the robot's Wi-Fi network.
2. Have an E-Stop active and within reach.
3. Make sure no other client holds the robot's **body lease**. For example,
   release control on the tablet. Each script takes the lease when it starts
   and returns it when it exits.

### Step 1: check the connection

```powershell
python scripts/basic_commands.py
```

The robot powers on and shows a menu (`1` Sit, `2` Stand, `0` Quit). When you
quit, the robot powers off safely.

### Step 2: check the cameras and detection

```powershell
python scripts/grab_frame.py
```

This prints a table of camera sources, mount angles and person counts. It
saves the upright, annotated images to `captures/` and shows them on screen.
Press any key in an image window to close them.

### Step 3: run the follow-me demo

```powershell
python scripts/follow_me.py
```

The robot powers on. If it was already standing it stays standing; otherwise
it stays sitting. The **Follow Me** control window opens, along with one
preview window per camera.

| Control | Action |
| --- | --- |
| **Sit** / **Stand** | Change posture. Stand is disabled while Follow-Me is on. |
| **Pitch** | Toggle a nose-up "looking up" pose while standing idle. It is disabled in Follow-Me, which always keeps the body level. |
| **Follow-Me: ON/OFF** | Start or stop following. Turning it on stands the robot up if needed. |
| **◀ Left / Right ▶** | Move the lock to the next detected person on that side. |
| **Detection confidence** slider | Change the YOLO threshold while the demo runs. |
| **STOP** (red) | Stop all motion right away and turn Follow-Me off. |
| Close window / `q` in a preview | Stop, power off safely and exit. |

In the previews, the locked person is drawn in **green** and other people in
**grey**, with each person's track ID and distance.

When Follow-Me is on and no one is in view, Spot stands still and keeps
looking. If it loses the locked person for more than 8 frames, it locks onto
the nearest person instead, preferring anyone near where the lost person was
last seen.

---

## Safety

This software moves a heavy legged robot near people. Before using it:

- Run it only in an open, clear area, and keep an E-Stop in hand at all
  times.
- Keep in mind that the robot drives toward whoever it is locked onto, and
  that can change if detection fails. Watch the preview windows.
- The `MIN_SAFE_DISTANCE` floor and Spot's built-in obstacle avoidance
  reduce risk but do not remove it.
- Follow Boston Dynamics' safety guidelines and your institution's rules for
  operating robots.

---

## Project structure

```
.
├── scripts/
│   ├── basic_commands.py   # sit/stand menu (connection check)
│   ├── grab_frame.py       # one-shot capture + detection (camera check)
│   └── follow_me.py        # follow-me demo
├── assets/
│   └── santanna_logo.png   # logo used in this README
├── requirements.txt
├── LICENSE
└── README.md

# created at runtime (git-ignored)
├── spot-env/               # virtual environment
├── yolo11n.pt              # YOLO11n weights (auto-downloaded)
├── captures/               # grab_frame.py output
└── logs/                   # follow_me.py JSON-lines session logs
```

---

## Troubleshooting

- **`ResourceAlreadyClaimedError` / lease errors.** Another client (often the
  tablet) holds the body lease. Release control there and try again.
- **The robot won't power on.** Check that an E-Stop endpoint is registered
  and not triggered.
- **`none of the configured cameras (with depth) exist on this robot`.** The
  robot does not offer the expected `*_fisheye_image` and
  `*_depth_in_visual_frame` sources. Check the robot's software version.
- **The demo runs slowly.** Check that `torch.cuda.is_available()` is `True`
  (see installation step 5). Set `SHOW_WINDOWS = False` to skip drawing the
  preview windows.
- **`No module named 'tkinter'`.** On Linux, install `python3-tk`.

---

## License

Released under the [MIT License](LICENSE). Copyright © 2026 Dimitrios Tsiakmakis.

Spot and the Spot SDK are products of Boston Dynamics. This project is not
affiliated with or endorsed by Boston Dynamics.
