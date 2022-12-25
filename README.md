# CaRE

## Overview
<img src="https://user-images.githubusercontent.com/73362969/209476314-638d3e1a-256d-4f74-91e5-883ba8170577.png" width="400" height="300">

## Abstract
Robotic systems have several subsystems that possess a huge combinatorial configuration space and hundreds or even thousands of possible software and hardware configuration options interacting non-trivially. The configurable parameters can be tailored to target specific objectives, but when incorrectly configured, can cause functional faults. Finding the root cause of such faults and understanding the performance behavior is extremely challenging due to the vast and variable space, and the dependencies with the robots’ configuration settings and performance. This paper proposes CARE, a method for diagnosing the root cause of the functional faults through the lens of causality which abstracts the effects of environment configurations (e.g., obstacles) on robotic systems. We demonstrate CARE's efficacy by evaluating the diagnosed root cause of the functional faults, conducting experiments both in physical robots(Husky, and Turtlebot-3) and simulator (Husky). Furthermore, we demonstrate CARE's transferability reusing the causal performance model--- learned from the Husky simulator, for a different robotic system Turtlebot-3 physical platform).

# How to use Care
- Observational data collection: Record the oversvational data using [Reval](https://github.com/softsys4ai/Reval), currently supports `Husky` and `Turtlebot-3`
<p align="center">
  <img src= "https://user-images.githubusercontent.com/73362969/167684493-9181c890-4ec4-4503-8dc1-ba59fffc19e4.gif" width="500" height="300"/>
  <img src= "https://user-images.githubusercontent.com/73362969/209478527-f2ee23c1-532e-4fee-9a2f-30154cfb3d9c.gif" width="500" height="300"/>
   <em>Left: Husky-sim, Right: Turtlebot3-phy</em>
</p>  
- Run CARE: Using the given obervtional data

```python
# Traning
python run_care_training.py
# Inference
python run_care_inference.py
```
<p align="center">
  <img src= "https://user-images.githubusercontent.com/73362969/209478776-9d4e4f94-c525-4002-9ae0-4b1245266ca5.gif"/>
</p> 

## Customization
CARE can be applied to a different robotic system, given the observational data as `pandas.Dataframe`.

Example: Update the `run_care_training.py` as follows,

```python
    # read the observational
    df = pd.read_csv('observational_data')
    # read all columns
    columns = df.columns
    # Manipulable variables (e.g., configuration options)
    manipulable_variables = ['Cost_scaling_factor_global','Occdist_scale'] # replace with your own labels
    config_options = df[manipulable_variables]
    # Non-manipulable variables (e.g., evaluation metrics)
    non_manipulable_variables = ['Traveled_distance','Mission_time'] # replace with your own labels
    evaluation_metrics = df[non_manipulable_variables]  
    # Performance objective (e.g., energy, mission success)
    perf_objective = ['Battery_percentage','Mission_success'] # replace with your own labels
    objectives = df[perf_objective]
```
