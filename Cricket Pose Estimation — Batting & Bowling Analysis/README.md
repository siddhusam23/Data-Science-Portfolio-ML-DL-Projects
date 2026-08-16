# Cricket Pose Estimation — Batting & Bowling Analysis

A computer vision system that uses [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) and OpenCV to analyze cricket player biomechanics from video — tracking body landmarks in real time to extract technique metrics for **batters** and **bowlers**.

## Overview

The pipeline reads a video of a batting or bowling action, detects 33 body landmarks per frame using MediaPipe's pose model, and derives motion metrics from key joint positions to give quantitative feedback on technique.

**Current capabilities (Batting Analysis):**
- Real-time pose landmark detection and skeleton overlay
- Elbow angle calculation (shoulder–elbow–wrist) to assess bat-arm form
- Wrist speed tracking to measure swing velocity
- Swing phase detection — flags **Backswing** vs **Downswing** based on wrist speed and position
- Swing stability scoring — flags an **Unstable Swing** when wrist-speed variance exceeds a threshold over recent frames
- Live FPS counter for performance monitoring

**Planned (Bowling Analysis):**
- Bowling arm angle at release (legal-action check, front-arm elbow extension)
- Run-up speed and stride length tracking
- Release point consistency across deliveries
- Follow-through and body alignment metrics

## Repository Structure

```
cricket-pose-estimation/
├── src/
│   ├── batting_analysis.py     # pose estimation + swing metrics for batters
│   └── bowling_analysis.py     # (planned) pose estimation + action metrics for bowlers
├── videos/                     # sample input videos (not committed)
├── requirements.txt
├── .gitignore
└── README.md
```

## Metrics Explained

| Metric | Description |
|---|---|
| Elbow Angle | Angle at the elbow joint (shoulder–elbow–wrist), useful for checking bat-arm extension and bowling-arm legality |
| Wrist Speed | Frame-to-frame displacement of the wrist landmark, used as a proxy for swing/bowling-arm velocity |
| Swing Phase | Backswing/Downswing classification based on wrist speed thresholds and relative wrist–elbow position |
| Swing Stability | Standard deviation of wrist speed over the last 15 frames — high variance flags an inconsistent swing |
| FPS | Frames processed per second, for performance benchmarking |

## Setup

```bash
git clone https://github.com/<your-username>/cricket-pose-estimation.git
cd cricket-pose-estimation
pip install -r requirements.txt
```

## Usage

Place your input video in the `videos/` folder, update the video path in the script, and run:

```bash
python src/batting_analysis.py
```

Press `q` to quit the live analysis window.

## Tech Stack

- Python
- OpenCV
- MediaPipe Pose
- NumPy

## Roadmap

- [ ] Refactor into a reusable `PoseAnalyzer` class shared by batting and bowling scripts
- [ ] Add bowling action analysis module
- [ ] Export metrics to CSV/JSON for post-session review
- [ ] Support webcam input for live coaching sessions
- [ ] Add per-shot/per-delivery summary reports

## License

Released under the [MIT License](LICENSE).
