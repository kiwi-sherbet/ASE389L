# ASE893L: Two-wheeled robot
This is a light environment based on PyBullet physics simulation for a two-wheeled robot navigation. 
For more details, please see [this slide](https://docs.google.com/presentation/d/1K-S_7-c0nJ18lrR_ERDL0M3GuVp2w6TyM6yPqtzh53Q/edit#slide=id.gec8f79e481_0_90) presented in the class.

## Installation
Please install on a virtual environment.
Install dependencies by running the following commands:

```
pip3 install -r requirements.txt
```

## Usage
You can train the model with the following commands:
```
python script/train.py 
```

You can also test the trained policy with the following commands (need some modification for choosing the saved policy path):
```
python script/evaluate.py
```

You can train the gait-planning network with the following commands:
```
cd src
python3 train.py -v=VISUALIZATON_MODE -t=TERRAIN_TYPE -o=OBJECT_TYPE -q=ROBOT_TYPE
```

## Robot Models
You can modify the robot model in the simulation. You should not change the names of the links and the joints, or the code may not work. Check the following directory: `/data`

## Configuration Files
You can change the setup for rewards and PPO parameters. Check the following directory: `/config`

## Training Outputs
You can save logs and checkpoints. Check the following directory: `/save`
