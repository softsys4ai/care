# CaRE

## Overview
<img src="https://user-images.githubusercontent.com/73362969/209476314-638d3e1a-256d-4f74-91e5-883ba8170577.png" width="600" height="500">

## Abstract
Robotic systems have several subsystems that possess a huge combinatorial configuration space and hundreds or even thousands of possible software and hardware configuration options interacting non-trivially. The configurable parameters can be tailored to target specific objectives, but when incorrectly configured, can cause functional faults. Finding the root cause of such faults and understanding the performance behavior is extremely challenging due to the vast and variable space, and the dependencies with the robots’ configuration settings and performance. This paper proposes CARE, a method for diagnosing the root cause of the functional faults through the lens of causality which abstracts the effects of environment configurations (e.g., obstacles) on robotic systems. We demonstrate CARE's efficacy by evaluating the diagnosed root cause of the functional faults, conducting experiments both in physical robots(Husky, and Turtlebot-3) and simulator (Husky). Furthermore, we demonstrate CARE's transferability reusing the causal performance model--- learned from the Husky simulator, for a different robotic system Turtlebot-3 physical platform).

# How to use Care
- Observational data collection: Record the oversvational data using [Reval](https://github.com/softsys4ai/Reval), currently supports `Husky` and `Turtlebot-3`
<p align="center">
  <img src= "https://user-images.githubusercontent.com/73362969/167684493-9181c890-4ec4-4503-8dc1-ba59fffc19e4.gif" width="500" height="300"/>
  <img src= "https://user-images.githubusercontent.com/73362969/167684493-9181c890-4ec4-4503-8dc1-ba59fffc19e4.gif" width="500" height="300"/>
</p>  
