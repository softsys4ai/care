[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7529716.svg)](https://doi.org/10.5281/zenodo.7529716)
# Care
- [Paper](https://ieeexplore.ieee.org/abstract/document/10137745)
- [Presentation](https://youtu.be/OomZr99hEKI?si=0t2b5IoywIdEYjEn)

## Overview
<img src="https://user-images.githubusercontent.com/73362969/209476314-638d3e1a-256d-4f74-91e5-883ba8170577.png" width="400" height="300">

## Abstract
Robotic systems have several subsystems that possess a huge combinatorial configuration space and hundreds or even thousands of possible software and hardware configuration options interacting non-trivially. The configurable parameters can be tailored to target specific objectives, but when incorrectly configured, can cause functional faults. Finding the root cause of such faults and understanding the performance behavior is extremely challenging due to the vast and variable space, and the dependencies with the robots’ configuration settings and performance. This paper proposes CARE, a method for diagnosing the root cause of the functional faults through the lens of causality which abstracts the effects of environment configurations (e.g., obstacles) on robotic systems. We demonstrate CARE's efficacy by evaluating the diagnosed root cause of the functional faults, conducting experiments both in physical robots(Husky, and Turtlebot-3) and simulator (Husky). Furthermore, we demonstrate CARE's transferability reusing the causal performance model--- learned from the Husky simulator, for a different robotic system Turtlebot-3 physical platform).

## Installation
```sh
git clone https://github.com/softsys4ai/care.git
cd ~/care && pip install -r requirements.txt
```

## How to use Care
- Observational data collection: Record the oversvational data using [Reval](https://github.com/softsys4ai/Reval), currently supports `Husky` and `Turtlebot-3`

<p align="center">
  <img src= "https://user-images.githubusercontent.com/73362969/167684493-9181c890-4ec4-4503-8dc1-ba59fffc19e4.gif" width="500" height="250"/> <img src= "https://user-images.githubusercontent.com/73362969/209478527-f2ee23c1-532e-4fee-9a2f-30154cfb3d9c.gif" width="500" height="250"/>
</p>    

- Run CARE: Using the given obervtional data

```python
# Traning
python run_care_training.py
# Inference
python run_care_inference.py
```
Buggy behaviors/Functional faults:

<img src= "https://user-images.githubusercontent.com/73362969/209481188-f7aadbd1-6505-4250-8e55-52582c887c25.gif" width="200" height="100"/> <img src= "https://user-images.githubusercontent.com/73362969/209481283-ab2a1c2d-6c26-4da3-94b9-1c7f6f048342.gif" width="200" height="100"/> <img src="https://user-images.githubusercontent.com/73362969/209482130-e31f71bb-2c70-4754-8e5d-8bcf289fc8fe.gif" width="200" height="100"/> <img src="https://user-images.githubusercontent.com/73362969/209482202-a13922c6-4406-48b4-adc5-c1ea6122621c.gif" width="200" height="100"/>
<p align="center">
<img src="https://user-images.githubusercontent.com/73362969/209482635-1245e414-c0c9-45df-96d7-331b0c2fee97.gif" width="200" height="110"/> <img src="https://user-images.githubusercontent.com/73362969/209482652-7ef091b1-d9bb-4ee5-a3af-5e2d37cb0603.gif" width="200" height="105"/>
</p> 
Diagnosing the root causes of the functional faults using CARE:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/73362969/209478776-9d4e4f94-c525-4002-9ae0-4b1245266ca5.gif"/>
</p> 

## Experiments
- Root-cause Verification Experiment: We first train the causal model using the observational data, and compute the ranks of the causal paths (the path’s ranks are provided in the `./care/result/rank_path.csv` file). We conduct 50 trials for each rank and recorded the energy, mission success, and evaluation metrics both in Husky simulator and physical robot. We provide the result of the trails in the `./care/result/exp` directory. To reproduce the results, we provide several functions in the `care_rootcause_viz.py` script.
- ransferability Experiment: We reuse the causal model constructed from the Husky simulator to determine the root-cause of the functional faults in the Turtlebot-3 physical robot. The list of root causes for different ranks are printed in the terminal along with the accuracy, precision, and recall. To reproduce the results, we provide the `care_transferibility_viz.py` script that produces the RMSE plot in the `./care/fig directory`.

## Customization
CARE can be applied to a different robotic system, given the observational data as a `pandas.Dataframe`.

Example: Update the `run_care_training.py` as follows,

```python
    # read the observational data
    df = pd.read_csv('observational_data.csv') # replace with your csv file
    # read all columns
    columns = df.columns
    # Manipulable variables (e.g., configuration options)
    manipulable_variables = ['Cost_scaling_factor_global','Occdist_scale'] # replace with your own labels
    # Non-manipulable variables (e.g., evaluation metrics)
    non_manipulable_variables = ['Traveled_distance','Mission_time'] # replace with your own labels
    # Performance objective (e.g., energy, mission success)
    perf_objective = ['Battery_percentage','Mission_success'] # replace with your own labels
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
}
```

## Contacts
Please feel free to contact via email if you find any issues or have any feedbacks. Thank you for using Care.
|Name|Email|     
|---------------|------------------|      
|Md Abir Hossen|mhossen@email.sc.edu|        
|Pooyan Jamshidi|pjamshid@cse.sc.edu|     

## 📘&nbsp; License
Care is released under the terms of the [MIT License](./LICENSE).

