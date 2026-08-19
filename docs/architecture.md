# System Architecture Specifications

## 1. High-Level System Architecture

The **Dynamic Accident Severity Estimation Model** is an end-to-end computer vision and web application system designed for autonomous traffic safety auditing and collision telemetry.

```text
[ Input Stream / Video Upload ]
              │
              ▼
[ Flask Video Processing Worker ]
              │
              ├──► [ YOLOv8 Accident Detector ] ────────┐
              ├──► [ ByteTrack Vehicle Tracker ] ──────┼──► [ Dynamic Severity Estimator Engine ]
              └──► [ YOLOv8 Fire/Smoke Detector ] ──────┘                 │
                                                                           ▼
                                                                [ Severity Score (0-100) ]
                                                                [ Minor / Major / Critical ]
                                                                           │
                                                                           ▼
                                                                  [ SQLite DB Storage ]
                                                                [ Snapshot & Video Clips ]
                                                                           │
                                                                           ▼
                                                                [ Flask Web Dashboard ]
```

## 2. Dynamic Severity Estimation Formula

The Dynamic Severity Index ($S \in [0, 100]$) is computed using multi-modal collision telemetry:

$$S = \min\left(100.0, \, S_{\text{impact}} + S_{\text{velocity}} + S_{\text{multi-vehicle}} + S_{\text{hazards}}\right)$$

### Factor Breakdown:
1. **Kinetic Impact Score ($S_{\text{impact}}$)**: Up to 30 points, derived from bounding box overlap and vehicle collision area compression ($0.0 - 30.0$).
2. **Velocity Drop Score ($S_{\text{velocity}}$)**: Up to 30 points, computed from deceleration magnitude ($\Delta v$ in km/h):
   $$S_{\text{velocity}} = \min\left(30.0, \frac{\Delta v}{60} \times 25.0\right)$$
3. **Multi-Vehicle & Mass Score ($S_{\text{multi-vehicle}}$)**: Up to 15 points, scaled by number of involved vehicles and vehicle mass category ($M_{\text{factor}}$: Motorcycle = 0.8, Sedan/Car = 1.0, SUV = 1.2, Bus = 1.6, Truck = 1.8):
   $$S_{\text{multi-vehicle}} = \min\left(15.0, (N_{\text{vehicles}} - 1) \times 7.5 \times M_{\text{factor}}\right)$$
4. **Hazard Multiplier Score ($S_{\text{hazards}}$)**:
   - Fire Detected: $+25.0$ points
   - Smoke Detected: $+10.0$ points

### Classification Categories:
- **Minor**: $S < 40.0$
- **Major**: $40.0 \le S < 75.0$
- **Critical**: $S \ge 75.0$
