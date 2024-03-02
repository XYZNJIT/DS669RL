### MDP Value Iteration and Policy Iteration
import random

from get_args import get_args
import numpy as np
import gymnasium as gym
import time

import warnings
warnings.filterwarnings('ignore')

np.set_printoptions(linewidth=np.inf)

np.set_printoptions(precision=3)


def interpret_policy(policy, nrow, ncol):
    """
    interpret a 2-D policy from number to action using: 0: L 1: D 2: R 3: U
    Parameters
    ----------
    policy
    nrow number of rows
    ncol number of columns

    Returns
    -------
    policy of each state with action first letter

    """
    policy = policy.reshape(nrow, ncol)
    re_policy = np.zeros((nrow, ncol), dtype=str)
    for i in range(len(policy)):
        for j in range(len(policy[i])):
            if policy[i][j] == 0:
                re_policy[i][j] = 'L'
            elif policy[i][j] == 1:
                re_policy[i][j] = 'D'
            elif policy[i][j] == 2:
                re_policy[i][j] = 'R'
            elif policy[i][j] == 3:
                re_policy[i][j] = 'U'
    return re_policy


def policy_evaluation(P, nS, policy, gamma=0.9, epsilon=1e-3, init_value=0.0):
    """
    Evaluate the value function from a given policy.
    :param P: transition probability
    :param nS: number of states
    :param policy: the policy to be evaluated
    :param gamma: gamma value for policy evaluation
    :param epsilon:
    :return:
    value_function: value function from policy evaluation
    evalution_steps: the number of steps need for policy evaluation
    """

    ############################
    # Initialize value function as all zeros
    # Modify the following line for initialization optimization in question 5.(a)
    # Hint: Please add a new parameter for the policy_iteration function and use this parameter to control the initialization.
    # value_function = np.zeros(nS)
    value_function = np.ones(nS) * init_value

    ############################
    # evaluation_steps: the number of steps needed for policy evaluation in each iteration
    evaluation_steps = 0

    ############################
    # Your Code #
    # Please use np.linalg.norm(x, np.inf) to calculate the infinity norm. #
    # Please use while loop to finish this part. #
    # Remember to update the evaluation_steps. #
    
    delta = np.inf
    while delta > epsilon:
        delta = 0
        for state in range(nS):
            old_value = value_function[state]
            action = policy[state]
            value_function[state] = sum([prob * (reward + gamma * value_function[next_state]) 
                                    for prob, next_state, reward, _ in P[state][action]])
            delta = np.linalg.norm([delta, abs(old_value - value_function[state])], np.inf)
        evaluation_steps += 1

    ############################
    return value_function, evaluation_steps


def policy_improvement(P, nS, nA, value_function, gamma=0.9):
    """
    Use the value function to improve the policy.
    :param P: transition probability
    :param nS: number of states
    :param nA: number of actions
    :param value_function: value function from policy iteration
    :param gamma: gamma value for policy improvement
    :return:
    new policy: An array of integers. Each integer is the optimal action to take in that state according to
                the environment dynamics and the given value function.
    """

    new_policy = np.zeros(nS, dtype="int")

    ############################
    # Your Code #
    # Please use np.argmax to select the best actions after getting the q value of each action. #

    for state in range(nS):
        q_values = np.zeros(nA)
        for action in range(nA):
            q_values[action] = sum([prob * (reward + gamma * value_function[next_state]) 
                        for prob, next_state, reward, _ in P[state][action]])
        new_policy[state] = np.argmax(q_values)
        
    ############################
    return new_policy


def policy_iteration(P, nS, nA, init_action=-1, gamma=0.9, epsilon=1e-3, init_value=0.0):
    """
    Runs policy iteration. Please call the policy_evaluation() and policy_improvement() methods to implement this method.
    :param P: transition probability
    :param nS: number of states
    :param nA: number of actions
    :param init_action: initial action for all the states, -1 for random action
    :param gamma:
    :param epsilon: epsilon parameter used in policy_evaluation()
    :return:
    value_function: np.ndarray[nS]
	policy: np.ndarray[nS]
	iteration: int, the number of iterations needed for policy iteration
    """

    value_function = np.zeros(nS)

    ############################
    # Initialize policy #
    # Your Code #
    # for the question of policy iteration initialization optimization
    init_policy = np.random.randint(0, nA, nS) if init_action == -1 else np.ones(nS, dtype=int) * init_action
    ############################

    # Number of iterations. The iteration does not include the steps of policy evaluation.
    iteration = 0

    # previous policy: the policy of last iteration.
    policy_prev = init_policy

    ############################
    # Your Code #
    # Please call the policy_evaluation() and policy_improvement() to update the policy. #
    # Remember to update the iteration and policy_prev. #
    # Please use while loop to finish this part. #
    total_evaluation_steps = 0
    
    terminate = False
    while not terminate:
        value_function, evaluation_steps = policy_evaluation(P, nS, policy_prev, gamma, epsilon, init_value)
        policy = policy_improvement(P, nS, nA, value_function, gamma)
        iteration += 1
        total_evaluation_steps += evaluation_steps
        print(f"For {iteration}th iteration, the evaluation steps are: {evaluation_steps}.")
        if np.array_equal(policy, policy_prev):
            terminate = True
        else:
            policy_prev = policy

    ############################

    print(f"The total number of evaluation steps is {total_evaluation_steps}.")
    print(f"There are {iteration} iterations in policy iteration.")
    return value_function, policy, iteration


def value_iteration(P, nS, nA, init_value=0.0, gamma=0.9, epsilon=1e-3):
    """
    Learn value function and policy by using value iteration method for a given gamma and environment.
    :param P: transition probability
    :param nS: number of states
    :param nA: number of actions
    :param init_value: initial value for value iteration
    :param gamma:
    :param epsilon: epsilon parameter used in value_evaluation()
    :return:
    value_function: np.ndarray[nS]
	policy: np.ndarray[nS]
	iteration: int, the number of iterations needed for value iteration
    """

    ############################
    # Initialize value #
    value_function = np.ones(nS) * init_value
    ############################

    # policy: the policy output from the generated value function after value iteration.
    policy = np.zeros(nS, dtype=int)
    iteration = 0

    ############################
    # Your Code #
    # Please use np.argmax to select the best action after getting the q value of each action. #
    # Please use np.linalg.norm(x, np.inf) to calculate the infinity norm. #
    # Please use while loop to finish this part. #


    while True:
        delta = 0
        for state in range(nS):
            q_values = np.zeros(nA)
            for action in range(nA):
                q_values[action] = sum([prob * (reward + gamma * value_function[next_state])
                            for prob, next_state, reward, _ in P[state][action]])
            max_value = np.max(q_values)
            delta = np.linalg.norm([delta, np.abs(max_value - value_function[state])], np.inf)
            value_function[state] = max_value
        iteration += 1
        # print(f"Value iteration: This is the {iteration}th iteration.")
        if delta < epsilon:
            break

    for state in range(nS):
        q_values = np.zeros(nA)
        for action in range(nA):
            q_values[action] = sum([prob * (reward + gamma * value_function[next_state]) 
                        for prob, next_state, reward, _ in P[state][action]])
        policy[state] = np.argmax(q_values)

    ############################

    # uncomment the following line if you need to print the value function
    # print('value_function:', value_function)

    print(f"There are {iteration} iterations in value iteration.")
    return value_function, policy, iteration


def render_single(env, policy, max_steps=100):
    """
    This function does not need to be modified
    Renders policy once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as
      attributes.
    Policy: np.array of shape [env.nS]
      The action to take at a given state
  """

    episode_reward = 0
    state, _ = env.reset()
    for t in range(max_steps):
        env.render()
        time.sleep(0.25)
        action = policy[state]
        state, reward, done, _, _ = env.step(action)
        episode_reward += reward
        if done:
            break
    env.render()
    if not done:
        print(f"The agent didn't reach a terminal state in {max_steps} steps.")
    else:
        print(f"Episode reward: {episode_reward}")


# Edit below to run policy and value iteration on different environments and
# visualize the resulting policies of actions.

if __name__ == "__main__":
    # get arguments from get_args.py
    args = get_args()

    # Initialize the gym environment and render
    env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", render_mode=args.render_mode, is_slippery=False)
    # Please check this link for the definition of state and actions of the FrozenLake game:
    # https://www.gymlibrary.dev/environments/toy\_text/frozen\_lake/

    # Number of state is 8 * 8 = 64
    env.nS = env.nrow * env.ncol
    # Number of action is 4
    env.nA = 4

    # Uncomment the following line to check and understand the format of the transition probability of FrozenLake.
    # print('transition probability:', env.P)

    # Running time start point
    start = time.time()
    # Initialize the average iteration
    avg_iteration = 0

    # Run the algorithm for "args.seeds" times. Each time with a different random seed.
    for i in range(args.seeds):

        # Reset the environment
        env.reset()
        # Set the random seed
        np.random.seed(i)
        random.seed(i)

        if args.method == 'policy_iteration':
            # Run policy iteration
            print("---- Policy Iteration----\n")
            value, policy, iteration = policy_iteration(
                env.P, env.nS, env.nA, init_action=args.init_action, gamma=args.gamma, epsilon=args.epsilon, init_value=args.init_value)

        elif args.method == 'value_iteration':
            # Run value iteration
            print("---- Value Iteration----\n")
            value, policy, iteration = value_iteration(
                env.P, env.nS, env.nA, init_value=args.init_value, gamma=args.gamma, epsilon=args.epsilon)
        else:
            raise ValueError('Unknown method')
        # Cumulate the number of iterations
        avg_iteration += iteration

        # Print the policy, interpreted to the actions' first letter (check the interpret_policy function).
        print('policy:', interpret_policy(policy.reshape(env.nrow, env.ncol), env.nrow, env.ncol))

    print('Total running time is:', time.time() - start,
          ' average number of iteration is:', avg_iteration / args.seeds)

    # Render the policy, the rendering do not require screenshots.
    render_single(env, policy, 100)
