import sys
from DQNModel import DQN  # A class of creating a deep q-learning model
from MinerEnv import (
    MinerEnv,
)  # A class of creating a communication environment between the DQN model and the GameMiner environment (GAME_SOCKET_DUMMY.py)
from Memory import (
    Memory,
)  # A class of creating a batch in order to store experiences for the training process
import numpy as np
from utils import plot_learning_curve


HOST = "localhost"
PORT = 1111
if len(sys.argv) == 3:
    HOST = str(sys.argv[1])
    PORT = int(sys.argv[2])


N_EPISODE = 100000  # The number of episodes for training
MAX_STEP = 1000  # The number of steps for each episode
BATCH_SIZE = 32  # The number of experiences for each replay

INITIAL_REPLAY_SIZE = 100  # The number of experiences are stored in the memory batch before starting replaying
INPUTNUM = 198  # The number of input values for the DQN model
ACTIONNUM = 6  # The number of actions output from the DQN model
MAP_MAX_X = 21  # Width of the Map
MAP_MAX_Y = 9  # Height of the Map
load_checkpoint = False
# Initialize a DQN model and a memory batch for storing experiences


DQNAgent = DQN(
    INPUTNUM,
    ACTIONNUM,
    batch_size=BATCH_SIZE,
    mem_size=50000,
    eps_min=0.1,
    replace=1000,
    eps_dec=1e-5,
    chkpt_dir="models/",
    algo="dqnagent",
    env_name="minerai",
    gamma=0.99,
    epsilon=1,
    lr=0.00001,
)
if load_checkpoint:
    DQNAgent.load_models()


# Initialize environment
minerEnv = MinerEnv(HOST, PORT)
minerEnv.start()


fname = (
    DQNAgent.algo
    + "_"
    + DQNAgent.env_name
    + "_lr"
    + str(DQNAgent.lr)
    + "_"
    + str(N_EPISODE)
    + "games"
)
figure_file = "plots/" + fname + ".png"

n_steps = -100
scores, eps_history, steps_array = [], [], []


# Training Process
# the main part of the deep-q learning agorithm

best_score = -100
for episode_i in range(0, N_EPISODE):
    try:
        mapID = np.random.randint(1, 6)
        posID_x = np.random.randint(MAP_MAX_X)
        posID_y = np.random.randint(MAP_MAX_Y)
        request = (
            "map" + str(mapID) + "," + str(posID_x) + "," + str(posID_y) + ",50,100"
        )

        minerEnv.send_map_info(request)
        minerEnv.reset()
        s = minerEnv.get_state()
        
        terminate = False
        maxStep = minerEnv.state.mapInfo.maxStep

        n_steps = 0
        total_reward = 0
        while not terminate:
            action = DQNAgent.choose_action(s)
            minerEnv.step(str(action))
            s_next = minerEnv.get_state()
            reward = minerEnv.get_reward()

            terminate = minerEnv.check_terminate()
            if not load_checkpoint and not terminate:
                DQNAgent.store_transition(s, action, reward, s_next, terminate)
                DQNAgent.learn()

            
            total_reward += reward
            s = s_next
            n_steps += 1

        scores.append(total_reward)
        steps_array.append(n_steps + 1)

        avg_score = np.mean(scores[-100:])
        print(
            "episode: ",
            episode_i,
            " average score %.5f" % avg_score,
            "best score %.5f" % best_score,
            "epsilon %.2f" % DQNAgent.epsilon,
            "steps",
            n_steps,
        )

        if avg_score > best_score:
            best_score = avg_score
            DQNAgent.save_models()
        eps_history.append(DQNAgent.epsilon)

    except Exception as e:
        import traceback

        traceback.print_exc()
        print("Finished.")
        break
plot_learning_curve(steps_array, scores, eps_history, figure_file)
