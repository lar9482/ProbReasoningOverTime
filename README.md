# 2nd project from CS 5313-01

Advanced Artificial Intelligence

Taught by Dr. Sandip Sen at the University of Tulsa in FA 23.

## This repository implements the following: 

#### Exact inference techniques for a Hidden Markov Model
  - Filtering P(S_t | E_1:t) and Smoothing using the Country-Dance Algorithm
  - Online Smoothing P(S_k | E_1:t) using Online Fixed-Lag Smoothing
  - Most likely state sequence using the Viterbi Algorithm

#### Approximation Inference techniques for a robot maze, modelled as a Dynamic Bayesian Network:
  - Particle filtering using likelihood weighting
  - Quick demo of a particle filter instance:
    * https://www.youtube.com/watch?v=LBG_BjENyro
    * The red circle represents the position of the robot
    * The shaded regions indicate the likelihood of the robot's location generated from the particle filter


## Quick Results:

#### Exact Inference:
   - Given fixed evidence, Country-Dance, Fixed Lag, and Viterbi are quite accurate. 
     *  P(S_t | E_1:t), P(S_k | E_1:t) for k < t, and the argmaxS_t P(S_t | E_1:t) match with each other
   - If there is random evidence, then the performance is these algorithms is not sufficient for accurate inference.

#### Approximate Inference:
  - Particle filtering is quite accurate for inference, as long as there are enough particles.
      - E.G A particle filter instance with 1000 particles for quite good for 20x20 and 30x30 mazes, given the environment allows for some degree of uncertainty.

For more information about the results, please see [**info**](https://github.com/lar9482/ProbReasoningOverTime/tree/main/info)
