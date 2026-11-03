import gymnasium as gym
import numpy as np
import gym_super_mario_bros as mario
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import cv2
from nes_py.wrappers import JoypadSpace

def main():
    env = gym.make('SuperMarioBros-1-1-v0', render_mode='rgb_array')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    state, info = env.reset()

    nes_ram = env.unwrapped.ram()
    print("Emulater running press q to quit")

    for frame in range(1000):
        # chooses a randome action from [0,6]
        action = int(env.action_space.sample())
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        player_state_hex = f"0x{nes_ram[0x000E]:02X}"

        frame_bgr = cv2.cvtColor(state, cv2.COLOR_RGB2BGR)
        # sacle from 256x240 to 512x480 with pixel art edges
        frame_large = cv2.resize(
            frame_bgr, (512, 480), interpolation=cv2.INTER_NEAREST
        )

        # dark header for ai actions
        header = np.zeros((60,512,3), dtype=np.uint8)
        line1 = (
            f"Frame: {frame:04d} | Action: {str(SIMPLE_MOVEMENT[action]):<18} "
            f"| Reward: {reward:+5.1f}"
        )
        line2 = (
            f"RAM X: {info['x_pos']:04d} | RAM Y: {info['y_pos']:03d} | "
            f"Coins: {info['coins']:02d} | Status: {info['status']} ({player_state_hex})"
        )

        cv2.putText(
            header, line1, (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 128), 1, cv2.LINE_AA
        )
        cv2.putText(
            header, line2, (10, 48),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1, cv2.LINE_AA
        )