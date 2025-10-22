[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7529716.svg)](https://doi.org/10.5281/zenodo.7529716)
# Care
- [Paper](https://ieeexplore.ieee.org/abstract/document/10137745)
- [Presentation](https://youtu.be/OomZr99hEKI?si=0t2b5IoywIdEYjEn)

## Overview
<img src="https://user-images.githubusercontent.com/73362969/209476314-638d3e1a-256d-4f74-91e5-883ba8170577.png" width="400" height="300">

## Abstract
Robotic systems have several subsystems that possess a huge combinatorial configuration space and hundreds or even thousands of possible software and hardware configuration options interacting non-trivially. The configurable parameters can be tailored to target specific objectives, but when incorrectly configured, can cause functional faults. Finding the root cause of such faults and understanding the performance behavior is extremely challenging due to the vast and variable space, and the dependencies with the robots’ configuration settings and performance. This paper proposes CARE, a method for diagnosing the root cause of the functional faults through the lens of causality which abstracts the effects of environment configurations (e.g., obstacles) on robotic systems. We demonstrate CARE's efficacy by evaluating the diagnosed root cause of the functional faults, conducting experiments both in physical robots(Husky, and Turtlebot-3) and simulator (Husky). Furthermore, we demonstrate CARE's transferability reusing the causal performance model--- learned from the Husky simulator, for a different robotic system Turtlebot-3 physical platform.

## Installation
```bash
git clone -b causal_model https://github.com/softsys4ai/care.git
uv sync
```
## How to cite
If you use Care in your research or the dataset in this repository please cite the following:
```
@ARTICLE{10137745,
  author={Hossen, Md Abir and Kharade, Sonam and Schmerl, Bradley and Cámara, Javier and O'Kane, Jason M. and Czaplinski, Ellen C. and Dzurilla, Katherine A. and Garlan, David and Jamshidi, Pooyan},
  journal={IEEE Robotics and Automation Letters}, 
  title={CaRE: Finding Root Causes of Configuration Issues in Highly-Configurable Robots}, 
  year={2023},
  volume={8},
  number={7},
  pages={4115-4122},
  doi={10.1109/LRA.2023.3280810}}
```

## Contacts
Please feel free to contact via email if you find any issues or have any feedbacks. Thank you for using Care.
|Name|Email|     
|---------------|------------------|      
|Md Abir Hossen|mhossen@email.sc.edu|        
|Pooyan Jamshidi|pjamshid@cse.sc.edu|     

## 📘&nbsp; License
Care is released under the terms of the [MIT License](./LICENSE).

