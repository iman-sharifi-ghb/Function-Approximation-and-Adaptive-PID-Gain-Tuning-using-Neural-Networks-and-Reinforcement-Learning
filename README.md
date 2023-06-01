# Function-Approximation-and-Adaptive-PID-Gain-Tuning-using-Neural-Networks-and-Actor-Critic-algorithm

**System Identification and Self-Tuning PID Control using NN and reinforcement learning**

In this project, we will aim to tune the PID controller gains adaptively using Actor-Critic method with the radial basis or guassian kernels. We assume we don't have an accurate model of the system and that is why we take the advantage of Neural Networks to estimate the dynamical model of the system and then use the achieved model to find the best PID gains using Actor-Critic Reinforcement Learning method.

## Notes

`PID_FA_NN.m` : 

This file is Fucntion approximation using Neural Networks with Adaptive PID Gains. You can read the attached `PID Neural Networks.pdf` file for learning the algorithm and structures.

![PID_FA_NN](https://user-images.githubusercontent.com/60617560/129597840-e8d9f399-4de6-4a1a-8218-b4fd27fd5570.png)

![PID_gains](https://user-images.githubusercontent.com/60617560/129597930-453bcfa4-9962-4000-905a-179b3a898e61.png)

`FA_A2C.m` :

    Function Approximation using Actor-Critic Algorithm
![FA_A2C](https://user-images.githubusercontent.com/60617560/129596768-e3680e6c-bc19-4833-b5cb-73681c8fb1ef.png)

If you want to change the dynamic system, Please just change the `NonLinDynamic(.)` function in MATLAB files.
