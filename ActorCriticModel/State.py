import numpy as np
import time
import hashlib
from SharedMemory import read_memory_and_split 

def hash_categorical(value):
    """Hashes a categorical value using SHA-256 and returns a fixed-size vector."""
    value_str = str(value)
    hash_object = hashlib.sha256(value_str.encode())
    hash_hex = hash_object.hexdigest()
    hash_vector = np.array([int(hash_hex[i:i+2], 16) for i in range(0, 16, 2)])
    return hash_vector

def preprocess_state(action_history):
    players_data = read_memory_and_split()  # Read player data from shared memory

    # Extract `IsRoundOver` if present
    is_round_over = players_data.pop('IsRoundOver', False)  # Default to False if not found
    
    state_vectors = []  # List to store state vectors for each player

    for player_number, player_data in players_data.items():
        # Hash the necessary categorical values
        movement_state_hash = hash_categorical(player_data['character']['MovementState'])
        attack_state_hash = hash_categorical(player_data['character']['AttackState'])
        inputs_hash = hash_categorical(player_data['input']['Inputs'])
        current_move_frame_hash = hash_categorical(player_data['frame']['CurrentMoveFrame'])
        current_move_reference_id_hash = hash_categorical(player_data['frame']['CurrentMoveReferenceID'])

        # Use the raw Facing value (1 or -1)
        facing = player_data['input']['Facing']  # This should be either 1 or -1
        move_hit = 1 if player_data['attack']['MoveHit'] else 0
        move_guarded = 1 if player_data['attack']['MoveGuarded'] else 0

        # Include the ComboCount
        combo_count = player_data['attack']['ComboCount']

        # Normalize percentage values (already assumed to be between 0 and 1)
        life_percentage = player_data['meter']['LifePercentage']
        meter_percentage = player_data['meter']['MeterPercentage']
        meter_max = player_data['meter']['MeterMax']
        dizzy_percentage = player_data['meter']['DizzyPercentage']
        guard_points_percentage = player_data['meter']['GuardPointsPercentage']
        recoverable_hp_percentage = player_data['meter']['RecoverableHpPercentage']

        # Include velocity values
        horizontal_velocity = player_data['velocity']['HorizontalVelocity']
        vertical_velocity = player_data['velocity']['VerticalVelocity']

        # Include position values
        position_x = player_data['position']['PositionX']
        position_y = player_data['position']['PositionY']

        # Get action history for this player and ensure it's a 1D array
        recent_actions = action_history[player_number]  # 1D array of recent actions

        # Ensure action history is of fixed length (e.g., pad with -1 if needed)
        max_history_length = 5  # Adjust as necessary
        if len(recent_actions) < max_history_length:
            recent_actions = np.pad(
                recent_actions, 
                (0, max_history_length - len(recent_actions)), 
                mode='constant', constant_values=-1
            )
        else:
            recent_actions = recent_actions[-max_history_length:]  # Take only the last N actions

        # Create a combined state vector for the player
        state_vector = np.concatenate([
            movement_state_hash,
            attack_state_hash,
            inputs_hash,
            current_move_frame_hash,
            current_move_reference_id_hash,
            np.array([facing, move_hit, move_guarded, combo_count]),  # Added combo_count here
            np.array([
                life_percentage,
                meter_percentage,
                meter_max,
                dizzy_percentage,
                guard_points_percentage,
                recoverable_hp_percentage,
                horizontal_velocity,
                vertical_velocity,
                position_x,
                position_y
            ]),
            recent_actions  # Include action history
        ])

        if life_percentage == 0:
            is_round_over = True

        state_vectors.append(state_vector)  # Add the state vector to the list
        # print(state_vectors)

    return np.array(state_vectors), is_round_over  # Return a numpy array of all state vectors

if __name__ == "__main__":
    while True:
        state_vector, _ = preprocess_state([[],[]])
        print(state_vector[1][52])
        time.sleep(1)