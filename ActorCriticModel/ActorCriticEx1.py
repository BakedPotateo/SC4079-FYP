import subprocess
import os
import pyautogui

import concurrent.futures

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import win32gui
import win32api
import win32con
import time

from Action import ActionSpace
from State import preprocess_state  # Function to preprocess game state

import matplotlib.pyplot as plt

# Store rewards for both agents
rewards_history = [[], []]  # One list per agent

# Initialize action_history: One list per player, initially empty
action_history = [[] for _ in range(2)]

attack_actions = [  # Basic Attacks
                    "LPH",  # light punch
                    "MPH",  # medium punch
                    "LKI",  # light kick
                    "MKI",  # medium kick
                    "CLP",  # crouching light punch
                    "CMP",  # crouching medium punch
                    "CLK",  # crouching light kick
                    "CMK",  # crouching medium kick
                    "THR",  # throw
                    
                    # Supers
                    "TKP",  # Triple Kung-Fu Palm
                    "SKU",  # Smash Kung-Fu Upper
                    
                    # Specials
                    "UPX",  # DP + x
                    "UPY",  # DP + y
                    "UXY",  # DP + xy
                    "QFX",  # QCF + x
                    "QFY",  # QCF + y
                    "QXY",  # QCF + xy
                    "QBX",  # QCB + x
                    "QBY",  # QCB + y
                    "QBC",  # QCB + xy
                    
                    "QFA",  # QCF + a
                    "QFB",  # QCF + b
                    "QAB",  # QCF + ab
                    "FFA",  # Dash + a
                    "FFB",  # Dash + b
                    "FAB",   # Dash + ab
                ]

def run_ikemen_game(exe_path):
    """Run the Ikemen GO executable file."""
    print(f"Launching {exe_path}...")
    # Start the Ikemen GO executable
    process = subprocess.Popen([exe_path], cwd=os.path.dirname(exe_path))
    time.sleep(3)  # Wait for a few seconds to ensure the game has started properly
    return process

# Actor-Critic Neural Network
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        
        # Shared layers
        self.shared_fc1 = nn.Linear(state_size, 1024)
        self.shared_fc2 = nn.Linear(1024, 512)
        self.shared_fc3 = nn.Linear(512, 256)
        self.shared_fc4 = nn.Linear(256, 128)

        # Define separate actor and critic heads
        self.actor_fc = nn.Linear(128, action_size)  # Output matches the number of actions
        self.critic_fc = nn.Linear(128, 1)

    def forward(self, state):
        x = torch.relu(self.shared_fc1(state))
        x = torch.relu(self.shared_fc2(x))
        x = torch.relu(self.shared_fc3(x))
        x = torch.relu(self.shared_fc4(x))

        logits = self.actor_fc(x)  # Actor output
        state_value = self.critic_fc(x)  # Critic output

        return logits, state_value


def record_action(player_num, action, action_space):
    """Record the action taken by a player with fixed-size history."""
    global action_history

    # Get the integer encoding of the action
    action_int = action_space.action_to_int.get(action, -1)  # Use -1 for unknown actions

    # Add the new action to the player's history
    action_history[player_num].append(action_int)

    # Ensure the history is exactly 50 elements
    if len(action_history[player_num]) > 5:
        action_history[player_num].pop(0)  # Remove the oldest action


    
    
# Reward function to evaluate state changes
def calculate_reward(pNum, episode, current_state, previous_state, elapsed_time, action, round_over=False):
    """
    Calculate rewards based on changes in the game state between the previous and current states.
    
    Args:
    current_state (np.array): Current state vector for both players.
    previous_state (np.array): Previous state vector for both players.
    round_over (bool): Boolean indicating if the round has ended.

    Returns:
    reward (float): Calculated reward based on state differences.
    """
    # Extract relevant values for reward calculation
    agent_current = current_state[pNum]
    agent_previous = previous_state[pNum]
    opponent_current = current_state[1-pNum]
    opponent_previous = previous_state[1-pNum]

    # Updated indices based on the new `state_vector` structure
    combo_index = 43
    life_index = 44  # LifePercentage index in the concatenated state vector
    meter_index = 45  # MeterPercentage index in the concatenated state vector
    move_hit_index = 41  # MoveHit flag index in the concatenated state vector
    move_guarded_index = 42  # MoveGuarded flag index in the concatenated state vector
    guard_broken_index = 48  # GuardPointsPercentage index, if 0 indicates guard break
    position_x_index = 52  # PositionX index in the state vector
    position_y_index = 53
    
    # Extract health, meter, and other values
    combo_count = agent_current[combo_index]
    agent_health_change = agent_previous[life_index] - agent_current[life_index]
    opponent_health_change = opponent_previous[life_index] - opponent_current[life_index]

    agent_meter_change = agent_current[meter_index] - agent_previous[meter_index]
    opponent_meter_change = opponent_current[meter_index] - opponent_previous[meter_index]

    agent_hit = agent_current[move_hit_index] > agent_previous[move_hit_index]
    opponent_hit = opponent_current[move_hit_index] > opponent_previous[move_hit_index]

    agent_guarded = agent_current[move_guarded_index] > agent_previous[move_guarded_index]
    opponent_guarded = opponent_current[move_guarded_index] > opponent_previous[move_guarded_index]

    agent_guard_broken = agent_previous[guard_broken_index] > 0 and agent_current[guard_broken_index] == 0

    agent_x = agent_current[position_x_index]
    opponent_x = opponent_current[position_x_index]
    
    # Distance between players
    distance = abs(agent_x - opponent_x)
    
    # Reward calculation
    reward = 0
    if combo_count > 1:
        reward += min(2 ** combo_count, 10)
        print("Combo)")
    if opponent_health_change > 0:
        reward += opponent_health_change * 10
        print("Opponent HP change)")
    if agent_health_change > 0:
        reward -= agent_health_change * 10
        print("Agent HP change")
    if agent_meter_change > 0:
        reward += agent_meter_change * 10
        print("Agent meter change")
    if agent_hit:
        reward += 5
        print("Agent successful hit")
        if opponent_current[position_y_index] != 0:
            reward += 5
    if opponent_hit:
        reward -= 3
        print("Opponent successful hit")
    global attack_actions
    if action in attack_actions and opponent_health_change == 0:
        reward -= 1
        print("Agent whiffed hit")
    if agent_guarded:
        reward += 2
        print("Agent guarded attack")
    if opponent_guarded:
        reward -= 1
        print("Opponent guarded attack")
    if agent_guard_broken:
        reward -= 8
        print("Agent guard broken")
    # if distance < 140 and action not in attack_actions:
    #     reward += 1
    if distance >= 90:
        reward -= 1
    if 239 < agent_x < 271  or -271 < agent_x < -239:
        print("Agent in corner")
        reward -= 2
    if 239 < opponent_x < 271  or -271 < opponent_x < -239:
        reward += 2
        print("Opponent in corner")
    if round_over:
        if agent_current[life_index] > opponent_current[life_index]:
            reward += 10
            print("Agent won round")
        else:
            reward -= 10
            print("Agent lost round")
    if elapsed_time > 40:
        return max(min(reward - elapsed_time / 100, 10), -10)
    else:
        return max(min(reward, 10), -10)  # Clamp reward between -10 and 10


def plot_rewards(rewards_history):
    """Plot the rewards of both agents across episodes and display summary statistics."""
    
    # Convert to numpy arrays for easier calculations
    rewards_agent_0 = np.array(rewards_history[0])
    rewards_agent_1 = np.array(rewards_history[1])

    # Handle the case where no rewards are recorded
    if rewards_agent_0.size == 0 or rewards_agent_1.size == 0:
        print("No rewards recorded for one or both agents.")
        return

    # Calculate statistics
    print("Rewards Summary:")
    print(f"Agent 0 - Mean: {rewards_agent_0.mean():.2f}, Max: {rewards_agent_0.max()}, Min: {rewards_agent_0.min()}")
    print(f"Agent 1 - Mean: {rewards_agent_1.mean():.2f}, Max: {rewards_agent_1.max()}, Min: {rewards_agent_1.min()}")

    # Plotting the rewards for each agent
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_agent_0, label='Agent 0', color='blue')
    plt.plot(rewards_agent_1, label='Agent 1', color='red')

    # Adding labels, title, and legend
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Total Reward', fontsize=14)
    plt.title('Agent Rewards per Episode', fontsize=16)
    plt.legend(loc='upper left')

    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)

    # Display the plot
    plt.show()


# Function to run a single agent
def run_agent(agent_id, hwnd, action_space, action_to_key, device, state_vector, gamma, episode, start_time):
    """Function to run a single agent."""
    state_size = 118
    action_size = len(action_space.actions)

    # Create model and optimizer for this agent
    model = ActorCritic(state_size, action_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-6)

    episode_reward = 0
    last_action = None  # Variable to store the last action taken
    repetition_penalty = 0
    
    movement_actions = ["FWD", "FDS"]
    
    # Main game loop for the agent
    while True:
        # Check elapsed time
        elapsed_time = time.time() - start_time
        # print(f"Elapsed time: {elapsed_time}")
        if elapsed_time > 100:
            print(f"Episode {episode + 1} terminated due to time limit.")
            break  # Exit the loop if 100 seconds have passed

        # Prepare state input for the current agent
        facing = state_vector[agent_id][40]
        x_pos_0 = state_vector[0][52]
        x_pos_1 = state_vector[1][52]
        dist = abs(x_pos_0 - x_pos_1)
        
        combined_vector = np.concatenate((state_vector[agent_id], state_vector[1 - agent_id]))
        state_tensor = torch.FloatTensor(combined_vector).unsqueeze(0).to(device)

        # Get logits and state value from the Actor-Critic model
        logits, state_value = model(state_tensor)
        logits = logits - logits.mean()  # Subtract the mean to center around 0

        # print("Logits:", logits.detach().cpu().numpy())
        
        # Convert logits to action probabilities
        temperature = 5.0
        temperature = max(2.0, temperature * (0.99 ** episode))
        action_probs = torch.softmax(logits / temperature, dim=-1)

        # Choose action based on probabilities
        action_probs_np = action_probs.squeeze(0).detach().cpu().numpy()
        # print(action_probs_np)
        epsilon = max(0.1, 1.0 - episode * 0.01)  # Epsilon decay for exploration
        if dist > 90:
            # Filter action indices and probabilities for movement actions
            movement_indices = [i for i, a in enumerate(action_space.actions) if a in movement_actions]
            movement_probs = action_probs_np[movement_indices]

            # Normalize movement_probs to sum to 1
            movement_probs /= movement_probs.sum()

            # Choose an action based on the filtered movement probabilities
            action_idx = np.random.choice(movement_indices, p=movement_probs)
            action_space.release_keys(agent_id)
        else:
            # Filter out movement actions to choose from other actions
            non_movement_indices = [i for i, a in enumerate(action_space.actions) if a not in movement_actions]
            non_movement_probs = action_probs_np[non_movement_indices]

            # Normalize non_movement_probs to sum to 1
            non_movement_probs /= non_movement_probs.sum()

            # Choose an action based on the filtered non-movement probabilities
            if np.random.rand() < epsilon:
                action_idx = np.random.choice(non_movement_indices)  # Random action from non-movement actions
            else:
                action_idx = np.random.choice(non_movement_indices, p=non_movement_probs)  # Probabilistic action from non-movement actions
        
        action = action_space.actions[action_idx]

        # Apply the penalty if the action is repeated
        if action == last_action and action not in ["FWD", "BWD", "FDS", "BLK", "BLC", "CRC"]:
            repetition_penalty = -2  # Penalize repeated action
        else:
            repetition_penalty = 0
        
        last_action = action
        
        # Take the action for the current agent
        action_space.take_action(agent_id, action, facing, hwnd)
        record_action(agent_id, action, action_space)
        time.sleep(1/60)

        global action_history
        # Get new state and check if round is over
        new_state_vector, is_round_over = preprocess_state(action_history)

        # Calculate reward based on the new state
        reward = calculate_reward(agent_id, episode, new_state_vector, state_vector, elapsed_time, action, is_round_over)
        print(f"Reward for player {agent_id}: {reward}")
        episode_reward += reward + repetition_penalty

        # Update state for both agents
        combined_new_vector = np.concatenate((new_state_vector[agent_id], new_state_vector[1 - agent_id]))
        new_state_tensor = torch.FloatTensor(combined_new_vector).unsqueeze(0).to(device)

        # Calculate the value of the new state
        _, next_state_value = model(new_state_tensor)

        # Calculate advantage (TD error)
        advantage = reward + (gamma * next_state_value[0]) - state_value[0]

        # Compute loss
        actor_loss = -torch.log(action_probs[0, action_idx].clamp(min=1e-10)) * advantage  # Avoid log(0)
        critic_loss = advantage ** 2
        entropy_loss = -torch.sum(action_probs * torch.log(action_probs + 1e-10))
        loss = actor_loss + critic_loss - 0.01 * entropy_loss


        # Backpropagate and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the current state for the next turn
        state_vector = new_state_vector

        # # Break if the round is over
        if is_round_over:
            print(f"Round {episode + 1} over")
            if episode == 499:
                # After training is complete
                save_model(model, optimizer, agent_id)
            break
        

    return episode_reward

def run_two_agents(hwnd, action_space, action_to_key, device):
    gamma = 0.99
    for episode in range(500):
        print(f"Starting episode {episode + 1}")

        global action_history
        
        # Initialize states for both agents
        state_vector, is_round_over = preprocess_state(action_history)

        # Capture start time of the episode
        start_time = time.time()

        # Use ThreadPoolExecutor to run both agents in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            for agent_id in range(2):
                # Pass the episode number and start time to the run_agent function
                futures.append(executor.submit(run_agent, agent_id, hwnd, action_space, action_to_key, device, state_vector, gamma, episode, start_time))

            # Wait for both agents to complete and get their rewards
            rewards = [future.result() for future in futures]

        # Print episode results
        print(f"Episode {episode + 1}\nAgent 1 Reward: {rewards[0]}\nAgent 2 Reward: {rewards[1]}")

        # Store rewards for plotting later
        rewards_history[0].append(rewards[0])  # Agent 1 reward
        rewards_history[1].append(rewards[1])  # Agent 2 reward
        
        # Restart round after each episode
        if time.time() - start_time >= 100:
            action_space.macro(['shift', 'f4'])
        else:
            pyautogui.keyDown('f4')
        action_space.release_keys(0)
        action_space.release_keys(1)
        time.sleep(2)
    
    # After all episodes, plot the rewards
    plot_rewards(rewards_history)

def save_model(model, optimizer, agent_id):
    """Save the model and optimizer states."""
    filename = f"actor_critic_ex_1_model_{agent_id}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filename)


def load_model(model, optimizer, filename="actor_critic_ex_1_model_0.pth"):
    """Load the model and optimizer states."""
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()  # Set the model to evaluation mode (no gradients)


def run_test_agent(agent_id, hwnd, action_space, action_to_key, device, model):
    """Run a single agent using the trained model."""
    model.eval()  # Ensure the model is in evaluation mode

    state_vector, is_round_over = preprocess_state(action_history)  # Get the initial game state
    episode_reward = 0

    while not is_round_over:
        # Prepare the state input
        facing = state_vector[agent_id][40]
        combined_vector = np.concatenate((state_vector[agent_id], state_vector[1 - agent_id]))
        state_tensor = torch.FloatTensor(combined_vector).unsqueeze(0).to(device)

        # Run the model to get action probabilities
        with torch.no_grad():  # Disable gradient calculations for inference
            logits, _ = model(state_tensor)
        
        action_probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        print(action_probs)

        # Choose an action based on probabilities
        action_idx = np.random.choice(len(action_space.actions), p=action_probs)
        action = action_space.actions[action_idx]
        key = action_to_key.get(action, None)

        # Perform the action
        if key:
            action_space.take_action(agent_id, key, facing, hwnd)

        # Get the new game state
        new_state_vector, is_round_over = preprocess_state(action_history)

        # Calculate reward (optional if you just want to test without rewards)
        # reward = calculate_reward(new_state_vector, state_vector, elapsed_time=0)
        # episode_reward += reward

        # Update the state for the next step
        state_vector = new_state_vector

    print(f"Test run completed. Agent {agent_id} total reward: {episode_reward}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    
    action_space = ActionSpace()
    hwnd = action_space.find_window("Ikemen GO")
    
    mode = 0
    
    if hwnd:
        if mode == 0:
            # Run the two-agent training loop
            run_two_agents(hwnd, action_space, {action: action for action in action_space.actions}, device)
        else:
             # Create the model and optimizer (must match training setup)
            state_size = 118
            action_size = len(action_space.actions)
            model = ActorCritic(state_size, action_size).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

            # Load the trained model
            load_model(model, optimizer, filename="actor_critic_ex_1_model_0.pth")
            run_test_agent(0, hwnd, action_space, {action: action for action in action_space.actions}, device, model)