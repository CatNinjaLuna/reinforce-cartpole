# RL-CartPole: Comparing Policy Gradient Methods in PyTorch

This repository implements and compares two Deep Reinforcement Learning (DRL) approaches to solve the `CartPole-v1` environment from OpenAI Gymnasium:

1. REINFORCE - A classic policy gradient method
2. Actor-Critic - A hybrid approach combining policy and value-based methods

## Implementation Architecture

### 1. REINFORCE (Policy Gradient)

The REINFORCE algorithm belongs to the class of policy gradient methods in Deep RL. Instead of learning a value function to decide actions, it directly learns a parameterized policy to maximize expected cumulative reward.

**Key Components**:

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

### 2. Actor-Critic

The Actor-Critic algorithm combines policy-based and value-based methods:

-  **Shared Network Architecture**:

   -  Shared feature extraction layer (128 units)
   -  Policy head (actor) outputs action probabilities
   -  Value head (critic) estimates state values

-  **Advantage-Based Updates**:

   -  Uses the critic's value estimates as a baseline
   -  Reduces variance in policy updates
   -  Learns faster than pure policy gradient methods

-  **Loss Function**:
   -  Combined loss with both actor and critic components
   -  Actor loss guides policy optimization
   -  Critic loss improves value estimation

## How to Run

### Installation

```bash
pip install gymnasium pygame torch
pip install "numpy<2.0.0"  # Downgrade numpy for compatibility
pip install matplotlib
```

### Run the Training Script

```bash
python rl_cartpole.py
```

You'll be prompted to select which method to run:

1. REINFORCE (Policy Gradient) only
2. Actor-Critic only
3. Both methods with comparison

You can also choose whether to enable rendering to visualize the agent's behavior during training.

## Implementation Details

-  Policy networks with one hidden layer (128 units)
-  Adam optimizer with learning rate of 1e-2 for REINFORCE and 1e-3 for Actor-Critic
-  Reward normalization for training stability
-  Episode reward tracking with visual plotting
-  Early stopping when CartPole is considered "solved"
-  Rendering toggle for visualization

## Results and Analysis

### REINFORCE (Policy Gradient)

The REINFORCE algorithm successfully solved the CartPole-v1 environment after approximately 230 episodes:

![REINFORCE Learning Curve](https://github.com/CatNinjaLuna/reinforce-cartpole/raw/main/policy_gradient_training_curve.png)

**Training Progression**:

-  **Initial Phase (Episodes 0-50)**: Agent performs poorly with low, inconsistent rewards around 20-30 per episode, showing the random exploration phase
-  **Exploration Phase (Episodes 50-100)**: Performance begins to fluctuate with occasional spikes up to 100+, showing early policy improvements
-  **Learning Phase (Episodes 100-150)**: High variability with some episodes reaching 300+ reward, demonstrating effective learning but still unstable policy
-  **Intermediate Phase (Episodes 150-200)**: Multiple spikes of high performance followed by periods of lower performance, showing the stochastic nature of policy gradient methods
-  **Breakthrough Phase (Episodes 200+)**: Rapid improvement leading to maximum reward (500) consistently, indicating optimal policy discovery

The training curve shows classic policy gradient characteristics:

1. High variance throughout training (the "noisy" nature of policy gradient methods)
2. Non-linear improvement with periods of progress and regression
3. Sudden performance jumps once critical policy configurations are discovered

The agent eventually achieved the perfect score of 500 (the maximum possible in this environment), officially "solving" the environment according to OpenAI Gymnasium standards.

### Actor-Critic

The Actor-Critic implementation shows a significantly different learning pattern:

![Actor-Critic Learning Curve](https://github.com/CatNinjaLuna/reinforce-cartpole/raw/main/actor_critic_training_curve.png)

**Training Progression**:

-  **Initial Phase (Episodes 0-50)**: Rapid initial learning with early spikes in performance up to 70-80 reward
-  **Decline Phase (Episodes 50-200)**: Performance actually decreases and stabilizes at a much lower level (around 10-15 reward)
-  **Stabilization Phase (Episodes 200+)**: The agent maintains this low but stable performance for the remainder of training

This learning pattern reveals that our Actor-Critic implementation:

1. Initially discovers a somewhat effective policy
2. Then converges to a suboptimal local minimum
3. Fails to escape this suboptimal policy throughout extended training

The Actor-Critic agent did not solve the environment within the maximum episode limit.

### Comparison Analysis

When comparing both methods directly:

![REINFORCE vs Actor-Critic Comparison](https://github.com/CatNinjaLuna/reinforce-cartpole/raw/main/rl_methods_comparison.png)

The comparison reveals several key insights:

1. **Learning Stability**:

   -  REINFORCE shows high variance but eventually reaches optimal performance
   -  Actor-Critic shows lower variance but fails to reach high performance

2. **Convergence Speed**:

   -  Actor-Critic initially learns faster (first 20-30 episodes)
   -  REINFORCE takes longer but achieves much better final performance

3. **Exploration-Exploitation Balance**:

   -  REINFORCE maintains better exploration throughout training
   -  Actor-Critic appears to converge prematurely to a suboptimal policy

4. **Implementation Sensitivity**:
   -  The Actor-Critic implementation likely requires further hyperparameter tuning
   -  Possible issues include learning rate, advantage normalization, or the balance between actor and critic losses

This comparison demonstrates that while Actor-Critic methods theoretically offer advantages over pure policy gradient methods like REINFORCE, they can be more sensitive to implementation details and hyperparameter choices. In this specific implementation, REINFORCE clearly outperforms our Actor-Critic approach for the CartPole environment.

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
