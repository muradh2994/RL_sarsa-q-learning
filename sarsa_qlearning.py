import numpy as np
import matplotlib.pyplot as plt

# Public variable, action directions for each arm
ACTION_N = 0
ACTION_H = 1
ACTION_E = 2
ACTION_W = 3
action = [ACTION_N, ACTION_H, ACTION_E, ACTION_W]


# Stores the current Agent user information and the required actions in the game state (SARSA, Q Learning)
class User(object):

    # Constructor
    def __init__(self, policyType, e=0.0, alpha=0.0, gamma=0.0):
        self.alpha = alpha  # Step size
        self.gamma = gamma  # Discounted rate
        self.e = e  # greedy probability
        self.current_state = []  # Sets Users current state

        self.policy_type = str(policyType)

    # Return String
    def __str__(self):
        return self.policy_type

    # Action movement, using a e-greedy method
    def move(self, state, state_Array):
        # If, probability less than e, pick random action
        if np.random.rand() < self.e:
            a = np.random.choice(action)

        # Else, find maximum Q value
        else:
            qState = state_Array[state[0], state[1], :]
            maxAction_val = np.argmax(qState)  
            # If, multiple max values, pick random action
            action_Array = np.where(qState == np.argmax(qState))[0]
            if len(action_Array) == 0:
                a = maxAction_val
            else:
                a = np.random.choice(action_Array)

        return a



    # SARSA policy  
    def sarsa(self, state_Array, cliff):

        # Reset current state
        self.current_state = cliff.startState

        # Choose action from state
        select_Act = self.move(self.current_state, state_Array)

        reward = 0.0  # Cumulative reward

        # Take action a and observe r untill reaching goal state
        while self.current_state != cliff.goalState:
            # Observe r
            r = cliff.rewardArr[self.current_state[0]][self.current_state[1]][select_Act]
            reward += r

            # Observe s'
            state_Val = cliff.next_state[self.current_state[0]][self.current_state[1]][select_Act]

            # choose a' from s'
            action_Val = self.move(state_Val, state_Array)

            # State-Action, gamma*Q(s',a')
            qSPriAPri = self.gamma * state_Array[state_Val[0], state_Val[1], action_Val]

            # Update Q(s,a), SARSA
            state_Array[self.current_state[0], self.current_state[1], select_Act] += \
                self.alpha * (r + qSPriAPri - state_Array[self.current_state[0]][self.current_state[1]][select_Act])

            # s<-s',a<-a'
            self.current_state = state_Val
            select_Act = action_Val

        # Return reward, limited to -100
        return max(reward, -100)


    # Q-Learning policy  
    def qLearning(self, state_Array, cliff):
        self.current_state = cliff.startState
        reward = 0.0

        while self.current_state != cliff.goalState:
            # Choose action from state
            select_Act = self.move(self.current_state, state_Array)

             # Observe R
            r = cliff.rewardArr[self.current_state[0]][self.current_state[1]][select_Act]
            reward += r  # Cumulative reward

            # Observe s'
            state_Val = cliff.next_state[self.current_state[0]][self.current_state[1]][select_Act]

            # set Q(s,a), QLearning
            qSPriAPri = self.gamma * np.max(state_Array[state_Val[0], state_Val[1], :])

            state_Array[self.current_state[0], self.current_state[1], select_Act] += \
                self.alpha * (r + qSPriAPri - state_Array[self.current_state[0]][self.current_state[1]][select_Act])

            # S <- s'
            self.current_state = state_Val

        # Return reward, limited to -100
        return max(reward, -100)


# Cliff Walking environment
class Cliff(object):


    def __init__(self, height=4, length=12):

        # State array Q, length X width X all actions
        self.state_Array = np.zeros((height, length, len(action)))
        self.startState = [3, 0]
        self.goal_state = [3, 11]
        self.cliffArea = [[3, 1],[3, 2],[3, 3],[3, 4],[3, 5],[3, 6],[3, 7],[3, 8],[3, 9],[3, 10]]

        # Reward Array
        self.reward_Array = np.zeros_like(self.state_Array)
        # cliff rewards
        self.reward_Array[:, :, :] = -1.0  
        # if action is selected from current posifion to cliff area, reward -100
        self.reward_Array[2, 1:11, ACTION_H] = -100.0
        self.reward_Array[3, 0, ACTION_E] = -100.0

        # to find next state
        self.next_state = []

        #loop through all states
        for i_H in range(0, height):
            
            self.next_state.append([])
            #
            for j_Len in range(0, length):

                # Create Array for each action to store next state
                state_Act = [[0, 0], [0, 0], [0, 0], [0, 0]]

                # Set state position depending on direction
                state_Act[ACTION_N] = [max(i_H - 1, 0), j_Len]
                state_Act[ACTION_W] = [i_H, max(j_Len - 1, 0)]
                state_Act[ACTION_E] = [i_H, min(j_Len + 1, length - 1)]
                state_Act[ACTION_H] = [min(i_H + 1, height - 1), j_Len]

                # Set Cliff area, reset to start state if User moves to position
                if self.reward_Array[i_H][j_Len][ACTION_N] <= -100:
                    state_Act[ACTION_N] = self.startState
                if self.reward_Array[i_H][j_Len][ACTION_E] <= -100:
                    state_Act[ACTION_E] = self.startState
                if self.reward_Array[i_H][j_Len][ACTION_W] <= -100:
                    state_Act[ACTION_W] = self.startState
                if self.reward_Array[i_H][j_Len][ACTION_H] <= -100:
                    state_Act[ACTION_H] = self.startState

                # Add dictionary to Next state array
                self.next_state[-1].append(state_Act)



# Environment state to run all objects in a single iteration
class Env(object):

    def __init__(self, Users, cliff, episodes=1):
        self.Users = Users
        self.cliff = cliff
        self.epi = episodes


    # Run game
    def run(self):
        # Reward array
        reward = np.zeros((self.epi, len(self.Users)))

        # Create independent copy of Q array for both policy tests
        state_Sarsa = np.copy(self.cliff.state_Array)
        state_QLear = np.copy(self.cliff.state_Array)

        
        for i in range(self.epi):

            agent_count = 0
            for j in self.Users:
                if j.policy == "SARSA":
                    reward[i][agent_count] += j.sarsa(state_Sarsa, self.cliff)
                elif j.policy == "QLearning":
                    reward[i][agent_count] += j.qLearning(state_QLear, self.cliff)
                agent_count += 1

        for j in self.Users:
            if j.policy == "SARSA":
                j.qState = state_Sarsa
            elif j.policy == "QLearning":
                j.qState = state_QLear

        return reward


# Main function
if __name__ == "__main__":


    ALPHA = 0.1  # Step size
    GAMMA = 1.0  # Learning Rate
    e = 0.1  # Greedy probability

    # Users - SARSA and qLearning policy
    Users = [User("SARSA", e, ALPHA, GAMMA),
              User("QLearning", e, ALPHA, GAMMA)]

    # Cliff
    Cliff_height = 4  # Cliff height
    Cliff_weight = 12  # Cliff Width
    cliff = Cliff(Cliff_height, Cliff_weight)

    # Test Environment
    EPISODES = 500  # Number of episodes
    ITERATIONS = 10  # Number of iterations
    environment = Env(Users, cliff, EPISODES)

    # Run environment, looping for the number of iterations and average results
    rewards = np.zeros((EPISODES, len(Users)))
    for iter in range(0, ITERATIONS):
        print("Running for Iteration:", iter)
        rewards += environment.run()

    # Average rewards
    rewards /= ITERATIONS

    sucEpisodes = 10 
    # Smoothed rewards, set walues to -100 to drop the first 10 values
    smotth_reward = np.zeros_like(rewards)
    smotth_reward[:] = -100

    # Calculate the mean of the results from the previous 10 episodes
    for i in range(sucEpisodes, EPISODES):
        for j in range(len(Users)):
            sum = 0.0
            for k in range(sucEpisodes, 0, -1):
                sum += rewards[i - k][j]

            smotth_reward[i][j] = sum / sucEpisodes

    # Plot Data to graph
    plt.title("Cliffwalking - SARSA vs. Q-Learning, e="+str(e))
    plt.plot(smotth_reward)
    plt.ylabel('Sum of rewards during episode')
    plt.xlabel('Episode')
    plt.legend(Users, loc=4)
    plt.show()

    #plotting a moving average over the last 10 episodes
    plt.title("Cliffwalking - SARSA vs. Q-Learning, e="+str(e))
    plt.ylabel('Average of rewards during last 10 episode')
    plt.xlabel('Episode')
    plt.legend(Users, loc=4)
    plt.plot((smotth_reward/sucEpisodes)[490:500])
    plt.show()   

  