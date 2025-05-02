"""Tabular QL agent"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import framework
import utils
np.random.seed(0)

DEBUG = False

GAMMA = 0.5  # discounted factor
TRAINING_EP = 0.5  # epsilon-greedy parameter for training
TESTING_EP = 0.05  # epsilon-greedy parameter for testing
NUM_RUNS = 10
NUM_EPOCHS = 200
NUM_EPIS_TRAIN = 25  # number of episodes for training at each epoch
NUM_EPIS_TEST = 50  # number of episodes for testing
ALPHA = 1  # learning rate for training

ACTIONS = framework.get_actions()
OBJECTS = framework.get_objects()
NUM_ACTIONS = len(ACTIONS)
NUM_OBJECTS = len(OBJECTS)


# pragma: coderesponse template
def epsilon_greedy(state_1, state_2, q_func, epsilon):
    """Returns an action selected by an epsilon-Greedy exploration policy

    Args:
        state_1, state_2 (int, int): two indices describing the current state
        q_func (np.ndarray): current Q-function
        epsilon (float): the probability of choosing a random command

    Returns:
        (int, int): the indices describing the action/object to take
    """
    
    if np.random.rand() < epsilon:
        # choose random action
        action_index = np.random.choice(NUM_ACTIONS)
        object_index = np.random.choice(NUM_OBJECTS)
    else:
        # choose action with the highest Q-value
        q_values = q_func[state_1, state_2, :, :]
        flat = np.argmax(q_values)
        action_index, object_index = np.unravel_index(flat, q_values.shape)



    return (action_index, object_index)


# pragma: coderesponse end


# pragma: coderesponse template
def tabular_q_learning(q_func, current_state_1, current_state_2, action_index,
                       object_index, reward, next_state_1, next_state_2,
                       terminal):
    """Update q_func for a given transition

    Args:
        q_func (np.ndarray): current Q-function
        current_state_1, current_state_2 (int, int): two indices describing the current state
        action_index (int): index of the current action
        object_index (int): index of the current object
        reward (float): the immediate reward the agent recieves from playing current command
        next_state_1, next_state_2 (int, int): two indices describing the next state
        terminal (bool): True if this episode is over

    Returns:
        None
    """
    q_current = q_func[current_state_1, current_state_2, action_index, object_index]

    if terminal:
        td_target = reward
    else:
        q_next_max = np.max(q_func[next_state_1, next_state_2, :, :])
        td_target = reward + GAMMA * np.max(q_func[next_state_1, next_state_2, :, :])


    q_func[current_state_1, current_state_2, action_index, object_index] += ALPHA * (td_target - q_current)

    return None  # This function shouldn't return anything


# pragma: coderesponse end


# pragma: coderesponse template
def run_episode(for_training):
    """ Runs one episode
    If for training, update Q function
    If for testing, computes and return cumulative discounted reward

    Args:
        for_training (bool): True if for training

    Returns:
        None
    """
    global q_func
    epsilon = TRAINING_EP if for_training else TESTING_EP

    epi_reward = None
    # initialize for each episode
    if not for_training:
        epi_reward = 0.0
        discount = 1.0


    # At top of run_episode (after newGame):
    room_desc, quest_desc, terminal = framework.newGame()
    room_idx  = dict_room_desc[ room_desc ]
    quest_idx = dict_quest_desc[ quest_desc ]

    while not terminal:
        # ε–greedy now uses integer indices
        action_idx, object_idx = epsilon_greedy(room_idx, quest_idx, q_func, epsilon)

        # execute in the environment (framework.step_game still takes descs)
        next_room_desc, next_quest_desc, reward, terminal = framework.step_game(
            room_desc, quest_desc, action_idx, object_idx
        )

        # convert next‐descs → next‐indices
        next_room_idx  = dict_room_desc[ next_room_desc ]
        next_quest_idx = dict_quest_desc[ next_quest_desc ]

        if for_training:
            tabular_q_learning(
                q_func,
                room_idx, quest_idx,
                action_idx, object_idx,
                reward,
                next_room_idx, next_quest_idx,
                terminal
            )
        else:
            epi_reward += discount * reward
            discount   *= GAMMA

        # move to the next state (both descs and indices)
        room_desc, quest_desc     = next_room_desc, next_quest_desc
        room_idx, quest_idx       = next_room_idx,  next_quest_idx

    if not for_training:
        return epi_reward



# pragma: coderesponse end


def run_epoch():
    """Runs one epoch and returns reward averaged over test episodes"""
    rewards = []

    for _ in range(NUM_EPIS_TRAIN):
        run_episode(for_training=True)

    for _ in range(NUM_EPIS_TEST):
        rewards.append(run_episode(for_training=False))

    return np.mean(np.array(rewards))


def run():
    """Returns array of test reward per epoch for one run"""
    global q_func
    q_func = np.zeros((NUM_ROOM_DESC, NUM_QUESTS, NUM_ACTIONS, NUM_OBJECTS))

    single_run_epoch_rewards_test = []
    pbar = tqdm(range(NUM_EPOCHS), ncols=80)
    for _ in pbar:
        single_run_epoch_rewards_test.append(run_epoch())
        pbar.set_description(
            "Avg reward: {:0.6f} | Ewma reward: {:0.6f}".format(
                np.mean(single_run_epoch_rewards_test),
                utils.ewma(single_run_epoch_rewards_test)))
    return single_run_epoch_rewards_test

if __name__ == '__main__':
    # 1. Build state‐index dictionaries and set global sizes
    dict_room_desc, dict_quest_desc = framework.make_all_states_index()
    NUM_ROOM_DESC = len(dict_room_desc)
    NUM_QUESTS    = len(dict_quest_desc)

    # 2. Load the game
    framework.load_game_data()

    # 3. Run NUM_RUNS complete Q‐learning experiments
    epoch_rewards_test = []
    for run_idx in range(NUM_RUNS):
        # (Optionally reseed here for per‐run reproducibility)
        np.random.seed(run_idx)
        epoch_rewards_test.append(run())

    epoch_rewards_test = np.array(epoch_rewards_test)  # shape = (NUM_RUNS, NUM_EPOCHS)

    # 4. Convergence detection (now that epoch_rewards_test is populated)
    mean_per_epoch = np.mean(epoch_rewards_test, axis=0)
    window, tol    = 10, 0.01
    final_val      = mean_per_epoch[-1]
    lower, upper   = final_val - tol, final_val + tol

    converged_epoch = None
    for e in range(len(mean_per_epoch) - window + 1):
        block = mean_per_epoch[e : e + window]
        if np.all((block >= lower) & (block <= upper)):
            converged_epoch = e
            break

    if converged_epoch is not None:
        print(f"Convergence epoch: {converged_epoch}")
        print(f"Average reward at convergence: {mean_per_epoch[converged_epoch]:.3f}")
    else:
        print("No convergence detected under tol=", tol)

    # 5. Plot the learning curve
    x = np.arange(NUM_EPOCHS)
    fig, axis = plt.subplots()
    axis.plot(x, mean_per_epoch)
    axis.set_xlabel('Epochs')
    axis.set_ylabel('Reward')
    axis.set_title(f'Tabular QL — avg over {NUM_RUNS} runs')
    plt.show()
