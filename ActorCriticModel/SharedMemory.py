import mmap
import json
import time
import ctypes
from ctypes import wintypes

# Constants for shared memory and mutex
SHM_NAME = "Global\\MySharedMemory"
SHM_SIZE = 8192
MUTEX_NAME = "Global\\MySharedMemoryMutex"

# Load the Windows API for mutex management
kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

def read_shared_memory():
    try:
        # Open the named mutex
        mutex_handle = kernel32.OpenMutexW(0x001F0001, False, MUTEX_NAME)
        if not mutex_handle:
            raise Exception("Failed to open mutex.")

        # Wait to acquire the mutex
        kernel32.WaitForSingleObject(mutex_handle, 0xFFFFFFFF)

        # Read data from shared memory
        with mmap.mmap(0, SHM_SIZE, SHM_NAME) as hfile:
            data = hfile.read(SHM_SIZE).decode().strip('\x00')

        # Release the mutex after reading
        kernel32.ReleaseMutex(mutex_handle)
        kernel32.CloseHandle(mutex_handle)

        return data
    except Exception as e:
        print(f"Error reading shared memory: {e}")
        return ""

def parse_json(data):
    """
    Parses the given JSON string and returns the resulting Python object.
    """
    try:
        return json.loads(data)
    except json.JSONDecodeError as e:
        # print(f"Error decoding JSON: {e}")
        return {}

def split_data_by_player(parsed_data):
    """
    Splits the parsed data into separate player entries based on PlayerNumber.
    Returns a dictionary with player numbers as keys and player-specific data as values.
    Additionally includes the `IsRoundOver` field if present.
    """
    players_data = {}
    
    # Check if `IsRoundOver` is present in the parsed data
    if 'isRoundOver' in parsed_data:
        # Include `isRoundOver` status in the returned dictionary
        players_data['IsRoundOver'] = parsed_data['isRoundOver']
        if parsed_data['isRoundOver'] or parsed_data['isRoundOver'] == 'true':
            print(f"Round Over: {parsed_data['isRoundOver']}")  # Output the round status for debugging

    # Ensure the data structure contains a list of `states`
    if 'states' in parsed_data and isinstance(parsed_data['states'], list):
        for entry in parsed_data['states']:
            player_number = entry['character']['PlayerNumber']
            players_data[player_number] = entry
    else:
        print()
        # print("Unexpected data format. Expected a dictionary with a 'states' list.")

    return players_data



def read_memory_and_split():
        # Read the game data from shared memory using semaphore protection
        game_output = read_shared_memory()
        # print(f"Data read from shared memory: '{game_output}'")

        # Parse the JSON data
        parsed_data = parse_json(game_output)

        # Split data by player
        players_data = split_data_by_player(parsed_data)

        # Display data for each player
        # for player_number, player_data in players_data.items():
        #     print(f"Player {player_number} Data: {json.dumps(player_data, indent=4)}")
        
        
            
        return players_data

if __name__ == "__main__":
    while True:
        read_memory_and_split()
        time.sleep(5)

