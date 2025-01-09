# Mapping the Real World: Transforming Image Coordinates to Real-World Coordinates for Robotic Systems

## Overview
This project focuses on converting 2D image coordinates obtained from a camera into 3D real-world coordinates suitable for robotic systems. This transformation is crucial for enabling robots to navigate and manipulate objects accurately in their environments.

## Components
- **Calibration Images Folder:** Contains chessboard images used for camera calibration.
- **Python File:** A script for calibrating the camera and simulating the results using PyBullet.

## Objectives
- Develop a robotic system that accurately maps 2D image coordinates to 3D real-world coordinates.
- Enhance robot's understanding of spatial relationships and interactions with real-world objects.

## Installation
1. Clone the repository:
   - git clone https://github.com/NouraMaklad/Image-to-World-Coordinate-Transformation.git
2. Navigate to the project directory:
   - cd Image-to-World Coordinate Transformation
3. Install required packages:
   - pip install -r requirements.txt
## Usage
- **Camera Calibration:**
  - Run the calibration script:
      - python calibrate_and_simulate.py
  - Specify the path to the calibration images folder in the script.
- **Simulation**
  - The script will automatically launch a PyBullet simulation to visualize the calibration results.
## Performance
  - Achieved an accuracy improvement of 95% in spatial transformations.
  - Mean reprojection error of less than 0.5 pixels.
## Contributors
- Aya Elsheshtawy
- Hager Tamer
- Noura Maklad
