# Solving game with reinforcement learning



The main game is being developed in the file SnakeGameClass where it is being being converted to an object oriented game


## Main files

__Agent.py__ Contains the agent to solve the game giving a _action-space_ of 3 (straight, left, right)

__Agent_4_directions__ Contains the agent to solve giving a _action-space_ of 4 (left, right, up, down)

__SnakeGameClass.py__ Contains the environment for the 3 _action-space_ game

__SnakeGameClass_4Directions.py__ Contains the environment for the 4 _action-space_ game

__HelperClasses.py__ Contains various classes used to help simplify calculations and implementation along the way

__PlaySnakeGame.py__ If you just want to play Snake without the AI

__model.py__ Where the Neural networks are defined


__Data_analysis.ipynb__ Jupyter notebook for loading in the data from __data__ directory where all the data is saved



## How to run the

To train or test the snake models you should run either of the __agent__ files, where at the bottom in the if name == main block you specify if you should the train or the test function.

While the game is running you can press s, to toggle slow or speed mode, press x to toggle render, or press q to exit the session.

