# Multi-Angle-Goal-Miss-Detection - Basketball 

## Basketball Goal Detection Pipeline:
Implemented a basketball detection pipeline that classifies goals and missed shots using computer vision techniques on video frames from any basketball match. The pipeline focuses on the basketball net region as the ROI, extracts that patch, and accurately detects goals and misses with 98% accuracy.
## Strategies Used:
- Trained three separate models on different camera angles of basketball videos.
- Compared their performance with a single mixed model trained on videos from all angles, achieving robust and consistent detection across perspectives.

## Detecton Results:
<div style="display: flex; flex-direction: row;">
    <img src="https://github.com/user-attachments/assets/f14e145b-c427-4217-91aa-5328e205c55c" width="220" />
    <img src="https://github.com/user-attachments/assets/db7e5e32-c1bf-4f5e-8134-fe8ff1cdb742" width="220"/>
    <img src="https://github.com/user-attachments/assets/8e900309-9522-4133-b510-cb1bdbdb3dae" width="220" />
    <img src="https://github.com/user-attachments/assets/bd2d3830-4e68-4079-bdf1-cf4ca6251db7" width="220" />
    <img src="https://github.com/user-attachments/assets/e4d3b79c-cf08-44c4-912f-b0b0d49f23ab" width="220" />
    <img src="https://github.com/user-attachments/assets/37ba889b-54eb-484d-8b81-9f6db9e9a9a3" width="220" />
    <img src="https://github.com/user-attachments/assets/34d7f765-6ac4-4da9-80b8-bde502969a4f" width="220" />
</div>

## Testing the TFLite EfficientDet Model:
- All 3 MOdels are being tested on the test data of all 3 angled video frmaes of basket ball
- By putting different prediction thresholds, all 3 models were tested. A threshold of 0.7 was giving desireable results



