# Reinforce-CartPole: Deep RL with Policy Gradient in PyTorch

This repository implements a simple Deep Reinforcement Learning (DRL) agent using the REINFORCE algorithm — a classic policy gradient method — to solve the `CartPole-v1` environment from OpenAI Gymnasium.

## Deep Reinforcement Learning Architecture: REINFORCE

This project uses the REINFORCE algorithm, which belongs to the class of policy gradient methods in Deep RL. Instead of learning a value function to decide actions, REINFORCE directly learns a parameterized policy to maximize expected cumulative reward.

**Background**: REINFORCE is a classic algorithm in the policy gradient family of deep reinforcement learning (Deep RL) methods. Its main idea is to directly learn how to make decisions (a policy), without relying on learning a value function.

### Key Components

-  **Policy Network**:

   -  A 2-layer feedforward neural network with softmax output
   -  Inputs: 4D state (cart position, velocity, pole angle, angular velocity)
   -  Outputs: Probability distribution over discrete actions (left or right)

-  **Sampling and Training**:

   -  Agent samples actions from the softmax policy
   -  After each episode, it computes the discounted return for each timestep
   -  Uses the log-likelihood trick to compute gradients and update the policy via gradient ascent

-  **Learning Signal**:

   -  Updates are weighted by the reward-to-go (discounted returns)
   -  Gradients encourage actions that lead to higher future rewards

-  **Early Stopping**:
   -  Training stops automatically when the agent achieves an average reward of 195.0+ over 100 consecutive episodes
   -  This follows OpenAI Gym's definition of "solving" the CartPole environment

## How to Run

### Installation

```bash
pip install gymnasium pygame torch
pip install "numpy<2.0.0"  # Downgrade numpy for compatibility
```

### Run the Training Script

```bash
python reinforce_cartpole.py
```

### Implementation Details

The implementation includes:

-  Policy network with one hidden layer (128 units)
-  Adam optimizer with learning rate of 1e-2
-  Reward normalization for training stability
-  Episode reward tracking with visual plotting
-  Early stopping when CartPole is considered "solved"
-  Rendering toggle for visualization

## Results and Analysis

The REINFORCE algorithm successfully solved the CartPole-v1 environment after 221 episodes. Looking at the training results:

![REINFORCE Learning Curve](https://github.com/CatNinjaLuna/reinforce-cartpole/raw/main/training_curve.png)

### Training Progression:

-  **Initial Phase (Episodes 0-50)**: Agent performs poorly with low, inconsistent rewards around 20-30 per episode, indicating random exploration
-  **Exploration Phase (Episodes 50-100)**: Performance begins to fluctuate with occasional spikes up to 60-80, showing early policy improvements
-  **Learning Phase (Episodes 100-150)**: High variability with some episodes reaching 250+ reward, demonstrating effective learning but still unstable policy
-  **Stabilization Phase (Episodes 150-200)**: More consistent performance around 100+ reward with gradual improvement, as policy becomes more reliable
-  **Breakthrough Phase (Episodes 200+)**: Rapid improvement leading to maximum reward (500) consistently, indicating optimal policy discovery

The training curve shows classic reinforcement learning characteristics:

1. High variance during early learning phases (exploration-exploitation tradeoff)
2. Non-linear improvement with plateaus followed by sudden improvements
3. Critical threshold behavior where performance rapidly jumps to optimal once key policy configurations are discovered

By episode 221, the agent achieved the perfect score of 500 (the maximum possible in this environment) with a 100-episode moving average exceeding 195, officially "solving" the environment according to OpenAI Gymnasium standards. The final average reward of 196.95 in episode 221 demonstrates that the agent can consistently balance the pole for the maximum duration.

This implementation demonstrates that despite REINFORCE being one of the simplest policy gradient methods, it can efficiently solve the CartPole problem with proper hyperparameter settings (learning rate, gamma value) and implementation details like return normalization.

## References

-  Sutton & Barto - Reinforcement Learning: An Introduction  
   http://incompleteideas.net/book/the-book-2nd.html

-  OpenAI Gymnasium: CartPole-v1  
   https://www.gymlibrary.dev/environments/classic_control/cart_pole/

-  REINFORCE Algorithm Explained (Lil'Log)  
   https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html

-  PyTorch Documentation  
   https://pytorch.org/docs/stable/index.html

## Author

Carolina Li  
LinkedIn: https://www.linkedin.com/in/carolinalyh/  
GitHub: https://github.com/CatNinjaLuna

## License

This project is open-source and available under the MIT License.
