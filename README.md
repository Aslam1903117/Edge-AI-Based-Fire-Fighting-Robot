# Edge-AI-Based-Fire-Fighting-Robot

This repository contains the firmware, model code, and associated header files for an Edge-AI-based firefighting robot developed using a hybrid MLP-MobileNetV2 approach, deployed on an ESP32-CAM platform. The project, detailed in the thesis "Enhancing Fire Detection in Firefighting Robots Using Hybrid MLP-MobileNetV2 with Edge AI on ESP32-CAM" by Md. Aslam Hossain (submitted on June 18, 2025, at Rajshahi University of Engineering & Technology), focuses on improving fire detection reliability in low-connectivity environments, particularly in Bangladesh. The MobileNetV2 dataset, due to its large size, is not uploaded here but can be requested separately from the author. This README was last updated at 10:30 AM +06 on Monday, June 23, 2025.

## Overview

The code implements a real-time fire detection system using:
- **Sensors**: DHT22 (temperature/humidity), MQ2 (gas), and flame sensors.
- **Model**: A hybrid approach combining MLP for sensor data and MobileNetV2 for image processing, optimized with TensorFlow Lite Micro.
- **Hardware**: ESP32-CAM with a dual-core Xtensa LX6 processor and 520 KB SRAM.
- **Features**: Detects fire hotspots with approximately 90% reliability and processes 96x96 pixel images.

The repository supports further development, addressing limitations such as false positives/negatives, confidence score uncertainties, and hardware constraints.


**Note**: The MobileNetV2 training dataset (5130 images) is not included due to its large size. Please contact the author (Md. Aslam Hossain) for access to the dataset if needed.

## Installation

### Prerequisites
- **Hardware**: ESP32-CAM development board, DHT22, MQ2, and flame sensors.
- **Software**:
  - [ESP-IDF](https://docs.espressif.com/projects/esp-idf/en/latest/get-started/index.html) (Espressif IoT Development Framework) v4.4 or later.
  - [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers) (included in the code).
  - Git and a C compiler (e.g., gcc).
  - Python 3.7+ (for model conversion, if needed).

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Aslam1903117/Edge-AI-Based-Fire-Fighting-Robot.git
   cd Edge-AI-Based-Fire-Fighting-Robot
