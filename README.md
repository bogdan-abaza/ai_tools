# ai_tools

Installation instructions:
1. Navigate to Your src ROS2 Workspace with the bringup package of your Robot:
cd ~/ros2_ws/src

2. Clone the ai_tools Repository:
git clone https://github.com/bogdan-abaza/ai_tools.git

3. Install dependencies from requirements.txt (see below):
pip install -r requirements.txt

4. Install Dependencies:
cd ~/ros2_ws
rosdep update
rosdep install --from-paths src --ignore-src -r -y

5. Build the Workspace:
colcon build --packages-install ai_tools

6. Source the Workspace:
source ~/ros2_ws/install/setup.bash

7. Run 1st time or when you want only to collect data  - ai_tools deisable
ros2 launch ai_tools ai_covariance.launch.py enable_ai:=false
CSV logs are saved in ~/ros2_ws/install/ai_tools/share/ai_tools/dataset/

8. Build your training model  - select which logs you need from step 7, merged them train the model (model.joblib)

9. Copy your trained model into: ai_tools/ai_tools/models
9.1 Copy your trained model.joblib model into ai_tools/ai_tools/models
9.2 Update the name of the model in ai_covariance_node by replacing the actual one which is "ai_covariance_model_full_v6.joblib"
   
11. Rebuild and source (steps 5 and 6)
 
13. Run ai_tools with inference for updating localization covariances in real time with ai enable:
ros2 launch ai_tools ai_covariance.launch.py enable_ai:=true

