import argparse

# python codebase_PI_VI_1.py -method policy_iteration

def get_args():
    parser = argparse.ArgumentParser()

    # Method
    parser.add_argument('-method', type=str, choices=['policy_iteration', 'value_iteration'],
                        default='policy_iteration', help='policy iteration or value iteration')

    # Parameters
    parser.add_argument('-seeds', type=int, default=1,
                        help='random seeds, in range [0, seeds)')
    parser.add_argument('-epsilon', type=float, default=1e-3,
                        help='epsilon to control the convergence of iteration')
    parser.add_argument('-gamma', type=float, default=0.9,
                        help='gamma for decreasing the reward')

    # Initializations
    parser.add_argument('-init_value', type=float, default=0,
                        help='initial value for value iteration')
    parser.add_argument('-init_action', type=int, choices=[-1, 0, 1, 2, 3], default=-1,
                        help='initial action for all the states, -1 for random action')

    # Render mode
    parser.add_argument(
        "-render_mode",
        "-r",
        type=str,
        help="The render mode for the environment. 'human' opens a window to render. 'ansi' does not render anything.",
        choices=["human", "ansi"],
        default="human",
    )

    return parser.parse_args()


"""
    For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
    the parameters P, nS, nA, gamma are defined as follows:
    
        P: nested dictionary of a nested lists
            From gym.core.Environment
            For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
            tuple of the form (probability, next_state, reward, terminal) where
                - probability: float
                    the probability of transitioning from "state" to "next_state" with "action"
                    probability is always 1.0 for deterministic environments.
                - next_state: int
                    denotes the state we transition to (in range [0, nS - 1])
                - reward: int
                    either 0 or 1, the reward for transitioning from "state" to
                    "next_state" with "action"
                - terminal: bool
                  True when "next_state" is a terminal state (hole or goal), False otherwise
        nS: int
            number of states in the environment
        nA: int
            number of actions in the environment
        gamma: float
            Discount factor. Number in range [0, 1)
    """
