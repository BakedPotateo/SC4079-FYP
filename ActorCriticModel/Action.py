import mmap
import json
import numpy as np

import pyautogui
import pydirectinput
import win32gui
import win32con
import win32api
import time

import keyboard

from State import preprocess_state

pydirectinput.PAUSE = 1/60

# Define the action space
class ActionSpace:
    def __init__(self):
        # Movements
        self.actions = [
            # Basic movement
            "RGT",  # right
            "LFT",  # left
            "JMP",  # jump
            "FJP",  # forward jump
            "BJP",  # backward jump
            "CRC",  # crouch
            "FDS",  # forward dash
            "BDS",  # backward dash
            "DJP",  # double jump
            "HLD",  # hold position

            # Basic Attacks
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

            # Defensive Actions
            "BLK",  # block
            "BLC",  # block crouch
            "UKM",  # ukemi (recovery roll)

        ]
        
        # Map each action to a unique integer
        self.action_to_int = {action: idx for idx, action in enumerate(self.actions)}
        self.int_to_action = {idx: action for idx, action in enumerate(self.actions)}
        
        self.player_mapping = [
            {
                'left':  'l',
                'right': 'j', 
                'up':    'i', 
                'down':  'k', 
                'A':     'f', 
                'B':     'g',
                'C':     'h',
                'X':     'r', 
                'Y':     't', 
                'Z':     'y'
            },
                    
            {
                'left':  'left',
                'right': 'right', 
                'up':    'up', 
                'down':  'down', 
                'A':     'z', 
                'B':     'x',
                'C':     'c',
                'X':     'a', 
                'Y':     's', 
                'Z':     'd'
            }
        ]

    def sample(self):
        # Randomly select an action from the action space
        return np.random.choice(self.actions)

    def find_window(self, title):
        hwnd = win32gui.FindWindow(None, title)
        if hwnd == 0:
            print(f"Window '{title}' not found!")
            return None
        
        # print(f"Window '{title}' found!")
        return hwnd

    def bring_window_to_front(self, hwnd):
        win32gui.SetForegroundWindow(hwnd)
        
    def release_keys(self, pNum):
        pydirectinput.PAUSE = 0/60
        
        if pNum == 0:
            keys = ['l', 'j', 'i', 'k', 'r', 't', 'y', 'f', 'g', 'h', 'shift', 'f4']
        else:
            keys = ['left', 'right', 'up', 'down', 'a', 's', 'd', 'z', 'x', 'c', 'shift', 'f4']
        
        for key in keys:
            pydirectinput.keyUp(key)
        pydirectinput.PAUSE = 1/60

    def keypress(self, key):
        self.release_keys()
        # Hold the key down for the specified duration
        pydirectinput.keyDown(key)  # Press the key down
 
        
    def macro(self, keys):
        pydirectinput.PAUSE = 0
        for key in keys:
            pydirectinput.keyDown(key)
        pydirectinput.PAUSE = 1/60
        
        
    def adjust_action_for_orientation(self, action, facing, playerNum, cpu):
        """Convert 'forward'/'backward' actions to 'left'/'right' based on character orientation."""
        if cpu:
            direction = 1
        else:
            direction = -1
        if action in ["FWD", "BWD"] and playerNum == 0:
            if facing == direction:
                return "l" if action == "FWD" else "j"
            else:
                return "j" if action == "FWD" else "l"
            
        elif action in ["FWD", "BWD"] and playerNum == 1:
            if facing == direction:
                return "right" if action == "FWD" else "left"
            else:
                return "left" if action == "FWD" else "right"
            
    def take_action(self, playerNum, actionStr, facing, hwnd, cpu=False):
        print(f"Sending action {actionStr} to agent {playerNum}, facing {facing}.")

        keyMap = self.player_mapping[playerNum]
            
        self.bring_window_to_front(hwnd)
        match actionStr:
            # Movement actions
            case "FWD":
                pydirectinput.keyUp(keyMap['down'])
                dirKey = self.adjust_action_for_orientation("FWD", facing, playerNum, cpu)
                pydirectinput.keyDown(dirKey)
                # time.sleep(10/60)
                # pydirectinput.keyUp(dirKey)
            case "BWD":
                pydirectinput.keyUp(keyMap['down'])
                dirKey = self.adjust_action_for_orientation("BWD", facing, playerNum, cpu)
                pydirectinput.keyDown(dirKey)
                # time.sleep(10/60)
                # pydirectinput.keyUp(dirKey)
            case "JMP":
                self.release_keys(playerNum)
                pydirectinput.keyDown(keyMap['up'])
                self.release_keys(playerNum)
            case "FJP":
                self.release_keys(playerNum)
                dirKey = self.adjust_action_for_orientation("FWD", facing, playerNum, cpu)
                pydirectinput.keyDown(dirKey)
                pydirectinput.keyDown(keyMap['up'])
                # time.sleep(15/60)
                self.release_keys(playerNum)
            case "BJP":
                self.release_keys(playerNum)
                dirKey = self.adjust_action_for_orientation("BWD", facing, playerNum, cpu)
                pydirectinput.keyDown(dirKey)
                pydirectinput.keyDown(keyMap['up'])
                # time.sleep(15/60)
                self.release_keys(playerNum)
            case "CRC":
                pydirectinput.keyDown(keyMap['down'])
            case "FDS":
                pydirectinput.keyUp(keyMap['down'])
                dirKey = self.adjust_action_for_orientation("FWD", facing, playerNum, cpu)
                pydirectinput.keyDown(dirKey)
                pydirectinput.keyUp(dirKey)
                pydirectinput.keyDown(dirKey)
                time.sleep(10/60)
                # pydirectinput.keyUp(dirKey)
            case "BDS":
                pydirectinput.keyUp(keyMap['down'])
                dirKey = self.adjust_action_for_orientation("BWD", facing, playerNum, cpu)
                pydirectinput.keyDown(dirKey)
                pydirectinput.keyUp(dirKey)
                pydirectinput.keyDown(dirKey)
                self.release_keys(playerNum)
            case "DJP":
                pydirectinput.keyDown(keyMap['up'])
                time.sleep(8/60)
                pydirectinput.keyDown(keyMap['up'])
            case "HLD":
                self.release_keys(playerNum)
            
            # Basic Attack actions
            case "LPH":
                self.release_keys(playerNum)
                pydirectinput.keyDown(keyMap['X'])  # Light punch
                self.release_keys(playerNum)
                # time.sleep(2/60)
            case "MPH":
                self.release_keys(playerNum)
                pydirectinput.keyDown(keyMap['Y'])  # Medium punch
                self.release_keys(playerNum)
                # time.sleep(4/60)
            case "LKI":
                self.release_keys(playerNum)
                pydirectinput.keyDown(keyMap['A'])  # Light kick
                self.release_keys(playerNum)
                # time.sleep(2/60)
            case "MKI":
                self.release_keys(playerNum)
                pydirectinput.keyDown(keyMap['B'])  # Medium kick
                self.release_keys(playerNum)
                # time.sleep(4/60)
                
            case "CLP":
                pydirectinput.keyDown(keyMap['down'])
                pydirectinput.keyDown(keyMap['X'])
                pydirectinput.keyUp(keyMap['X'])
                self.release_keys(playerNum)
            case "CMP":
                pydirectinput.keyDown(keyMap['down'])
                pydirectinput.keyDown(keyMap['Y'])
                pydirectinput.keyUp(keyMap['Y'])
            case "CLK":
                pydirectinput.keyDown(keyMap['down'])
                pydirectinput.keyDown(keyMap['A'])
                pydirectinput.keyUp(keyMap['A'])
            case "CMK":
                pydirectinput.keyDown(keyMap['down'])
                pydirectinput.keyDown(keyMap['B'])
                pydirectinput.keyUp(keyMap['B'])
            case "THR":
                self.release_keys(playerNum)
                dirKey = self.adjust_action_for_orientation("FWD", facing, playerNum, cpu)
                pydirectinput.keyDown(dirKey)
                time.sleep(6/60)
                pydirectinput.keyDown(keyMap['Y'])
                time.sleep(4/60)
                self.release_keys(0)
                self.release_keys(1)
                # time.sleep(30/60)
            
            # Supers
            case "TKP":  # Triple Kung-Fu Palm
                self.release_keys(playerNum)
                dirKey = self.adjust_action_for_orientation("FWD", facing, playerNum, cpu)
                pydirectinput.keyDown(keyMap['down'])
                pydirectinput.keyDown(dirKey)
                pydirectinput.keyUp(keyMap['down'])
                pydirectinput.keyUp(dirKey)
                # time.sleep(1/60)
                pydirectinput.keyDown(keyMap['down'])
                pydirectinput.keyDown(dirKey)
                pydirectinput.keyUp(keyMap['down'])
                # time.sleep(1/60)
                pydirectinput.keyDown(keyMap['X']) # Light punch
                self.release_keys(playerNum)
                # time.sleep(60/60)
                
            case "SKU":  # Smash Kung-Fu Upper
                self.release_keys(playerNum)
                dirKey = self.adjust_action_for_orientation("BWD", facing, playerNum, cpu)
                pydirectinput.keyDown(keyMap['down'])
                pydirectinput.keyDown(dirKey)
                pydirectinput.keyUp(keyMap['down'])
                pydirectinput.keyUp(dirKey)
                time.sleep(1/60)
                pydirectinput.keyDown(keyMap['down'])
                pydirectinput.keyDown(dirKey)
                pydirectinput.keyUp(keyMap['down'])
                time.sleep(1/60)
                pydirectinput.keyDown(keyMap['X']) # Light punch
                self.release_keys(playerNum)
                # time.sleep(62/60)
            
            # Specials
            case "UPX":  # DP + x
                self.release_keys(playerNum)
                dirKey = self.adjust_action_for_orientation("FWD", facing, playerNum, cpu)
                pydirectinput.keyDown(dirKey)
                pydirectinput.keyUp(dirKey)
                pydirectinput.keyDown(keyMap['down'])
                pydirectinput.keyDown(dirKey)
                pydirectinput.keyDown(keyMap['X'])
                time.sleep(1/60)
                self.release_keys(playerNum)
                # time.sleep(60/60)

            case "UPY":  # DP + y
                self.release_keys(playerNum)
                dirKey = self.adjust_action_for_orientation("FWD", facing, playerNum, cpu)
                pydirectinput.keyDown(dirKey)
                pydirectinput.keyUp(dirKey)
                pydirectinput.keyDown(keyMap['down'])
                pydirectinput.keyDown(dirKey)
                time.sleep(1/60)
                pydirectinput.keyDown(keyMap['Y'])
                self.release_keys(playerNum)
                # time.sleep(60/60)

            case "UXY":  # DP + xy
                self.release_keys(playerNum)
                dirKey = self.adjust_action_for_orientation("FWD", facing, playerNum, cpu)
                pydirectinput.keyDown(dirKey)
                pydirectinput.keyUp(dirKey)
                pydirectinput.keyDown(keyMap['down'])
                pydirectinput.keyDown(dirKey)
                time.sleep(1/60)
                self.macro([keyMap['X'], keyMap['Y']])
                self.release_keys(playerNum)
                # time.sleep(60/60)
            
            case "QFX":  # QCF + x
                self.release_keys(playerNum)
                dirKey = self.adjust_action_for_orientation("FWD", facing, playerNum, cpu)
                pydirectinput.keyDown(keyMap['down'])
                pydirectinput.keyDown(dirKey)
                pydirectinput.keyUp(keyMap['down'])
                pydirectinput.keyUp(dirKey)
                time.sleep(1/60)
                pydirectinput.keyDown(keyMap['X'])
                self.release_keys(playerNum)
                # time.sleep(60/60)

            case "QFY":  # QCF + y
                self.release_keys(playerNum)
                dirKey = self.adjust_action_for_orientation("FWD", facing, playerNum, cpu)
                pydirectinput.keyDown(keyMap['down'])
                pydirectinput.keyDown(dirKey)
                pydirectinput.keyUp(keyMap['down'])
                pydirectinput.keyUp(dirKey)
                time.sleep(1/60)
                pydirectinput.keyDown(keyMap['Y'])
                self.release_keys(playerNum)
                # time.sleep(60/60)

            case "QXY":  # QCF + xy
                self.release_keys(playerNum)
                dirKey = self.adjust_action_for_orientation("FWD", facing, playerNum, cpu)
                pydirectinput.keyDown(keyMap['down'])
                pydirectinput.keyDown(dirKey)
                pydirectinput.keyUp(keyMap['down'])
                pydirectinput.keyUp(dirKey)
                time.sleep(1/60)
                self.macro([keyMap['X'], keyMap['Y']])
                self.release_keys(playerNum)
                # time.sleep(60/60)
            
            case "QBX":  # QCB + x
                self.release_keys(playerNum)
                dirKey = self.adjust_action_for_orientation("BWD", facing, playerNum, cpu)
                pydirectinput.keyDown(keyMap['down'])
                pydirectinput.keyDown(dirKey)
                pydirectinput.keyUp(keyMap['down'])
                pydirectinput.keyUp(dirKey)
                time.sleep(1/60)
                pydirectinput.keyDown(keyMap['X'])
                self.release_keys(playerNum)
                # time.sleep(60/60)

            case "QBY":  # QCB + y
                self.release_keys(playerNum)
                dirKey = self.adjust_action_for_orientation("BWD", facing, playerNum, cpu)
                pydirectinput.keyDown(keyMap['down'])
                pydirectinput.keyDown(dirKey)
                pydirectinput.keyUp(keyMap['down'])
                pydirectinput.keyUp(dirKey)
                time.sleep(1/60)
                pydirectinput.keyDown(keyMap['Y'])
                self.release_keys(playerNum)
                # time.sleep(60/60)
                
            case "QBC":  # QCB + xy
                self.release_keys(playerNum)
                dirKey = self.adjust_action_for_orientation("BWD", facing, playerNum, cpu)
                pydirectinput.keyDown(keyMap['down'])
                pydirectinput.keyDown(dirKey)
                pydirectinput.keyUp(keyMap['down'])
                pydirectinput.keyUp(dirKey)
                time.sleep(1/60)
                self.macro([keyMap['X'], keyMap['Y']])
                self.release_keys(playerNum)
                # time.sleep(60/60)
            
            case "QFA":  # QCF + a
                self.release_keys(playerNum)
                dirKey = self.adjust_action_for_orientation("FWD", facing, playerNum, cpu)
                pydirectinput.keyDown(keyMap['down'])
                pydirectinput.keyDown(dirKey)
                pydirectinput.keyUp(keyMap['down'])
                pydirectinput.keyUp(dirKey)
                time.sleep(1/60)
                pydirectinput.keyDown(keyMap['A'])
                self.release_keys(playerNum)
                # time.sleep(60/60)
                self.release_keys(playerNum)
            case "QFB":  # QCF + b
                self.release_keys(playerNum)
                dirKey = self.adjust_action_for_orientation("FWD", facing, playerNum, cpu)
                pydirectinput.keyDown(keyMap['down'])
                pydirectinput.keyDown(dirKey)
                pydirectinput.keyUp(keyMap['down'])
                pydirectinput.keyUp(dirKey)
                time.sleep(1/60)
                pydirectinput.keyDown(keyMap['B'])
                self.release_keys(playerNum)
                # time.sleep(60/60)
                
            case "QAB":  # QCF + ab
                self.release_keys(playerNum)
                dirKey = self.adjust_action_for_orientation("FWD", facing, playerNum, cpu)
                pydirectinput.keyDown(keyMap['down'])
                pydirectinput.keyDown(dirKey)
                pydirectinput.keyUp(keyMap['down'])
                pydirectinput.keyUp(dirKey)
                time.sleep(1/60)
                self.macro([keyMap['A'], keyMap['B']])
                self.release_keys(playerNum)
                # time.sleep(60/60)
            
            case "FFA":  # Dash + a
                self.release_keys(playerNum)
                dirKey = self.adjust_action_for_orientation("FWD", facing, playerNum, cpu)
                pydirectinput.keyDown(dirKey)
                pydirectinput.keyUp(dirKey)
                pydirectinput.keyDown(dirKey)
                time.sleep(1/60)
                pydirectinput.keyDown(keyMap['A'])
                self.release_keys(playerNum)
                # time.sleep(60/60)
                
            case "FFB":  # Dash + b
                self.release_keys(playerNum)
                dirKey = self.adjust_action_for_orientation("FWD", facing, playerNum, cpu)
                pydirectinput.keyDown(dirKey)
                pydirectinput.keyUp(dirKey)
                pydirectinput.keyDown(dirKey)
                time.sleep(1/60)
                pydirectinput.keyDown(keyMap['B'])
                self.release_keys(playerNum)
                # time.sleep(60/60)
                
            case "FAB":  # Dash + ab
                self.release_keys(playerNum)
                dirKey = self.adjust_action_for_orientation("FWD", facing, playerNum, cpu)
                pydirectinput.keyDown(dirKey)
                pydirectinput.keyUp(dirKey)
                pydirectinput.keyDown(dirKey)
                time.sleep(1/60)
                self.macro([keyMap['A'], keyMap['B']])
                self.release_keys(playerNum)
                # time.sleep(60/60)
        
            # Defensive actions
            case "BLK":  # Block (standing)
                dirKey = self.adjust_action_for_orientation("BWD", facing, playerNum, cpu)
                pydirectinput.keyDown(dirKey)
                # time.sleep(60/60)
                
            case "BLC":  # Block (crouching)
                dirKey = self.adjust_action_for_orientation("BWD", facing, playerNum, cpu)
                self.macro([keyMap['down'], dirKey])
                # time.sleep(60/60)
                
            case "UKM":  # Ukemi / Tech
                self.release_keys(playerNum)
                self.macro([keyMap['X'], keyMap['Y']])
                time.sleep(12/60)


if __name__ == "__main__":
    # Create an action space instance
    action_space = ActionSpace()
    print(action_space.player_mapping[1])
    window_title = "Ikemen GO"
    hwnd = action_space.find_window(window_title)
    
    state_vector, is_round_over = preprocess_state([[],[]])
    
    if hwnd:
        print("Press 'm' to exit the loop.")
        pNum = 0
        facing1 = state_vector[0][40]
        facing2 = state_vector[1][40]
        # Start a loop that will keep running until 'q' is pressed
        while not keyboard.is_pressed('m'):
            print(pNum)
            state_vector, is_round_over = preprocess_state([[],[]])
            facing1 = state_vector[0][40]
            facing2 = state_vector[1][40]
            action_space.take_action(0, "BLK", facing1, hwnd)
            # action_space.take_action(1, "FDS", facing2, hwnd)
            time.sleep(1/60)
            # Alternate between players
            pNum = 1 if pNum == 0 else 0
            # facing = -1 if facing == 1 else 1
            
        
        action_space.release_keys(0)
        action_space.release_keys(1)
        print("Exiting the loop. Program terminated.")