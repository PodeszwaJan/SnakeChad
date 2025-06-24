import os
import numpy as np
import pygame
from snake_game import SnakeGame
from dqn_agent import DQNAgent
from ddqn_agent import DDQNAgent
from adqn_agent import ADQNAgent # <-- MODIFIED
from excel_logger import ExcelLogger

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

def train_snake(render_mode=False, model_path=None, max_episodes=100, model_type="DQN", save_filename=None):
    game = SnakeGame(w=20, h=20, block_size=25, render_mode=render_mode)
    state_size = len(game._get_state())
    action_size = 3

    if save_filename is None:
        save_filename = f"{model_type.lower()}_model.pth" # <-- MODIFIED

    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    # --- MODIFIED: Agent creation block ---
    if model_type == "DDQN":
        agent = DDQNAgent(state_size, action_size)
    elif model_type == "ADQN":
        agent = ADQNAgent(state_size, action_size)
    else:  # Default to DQN
        agent = DQNAgent(state_size, action_size)
    
    scores = []
    best_score = 0
    logger = ExcelLogger() if not render_mode else None

    if model_path:
        full_model_path = os.path.join(MODELS_DIR, model_path)
        if os.path.exists(full_model_path):
            best_score = agent.load(full_model_path)
            print(f"Loaded {model_type} model from: {full_model_path} | Best score: {best_score}")
        else:
            print(f"Model file not found at {full_model_path}, starting from scratch.")
    else:
        print(f"Starting new training. Best model will be saved as '{save_filename}'.")

    for episode in range(1, max_episodes + 1):
        state = game.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, score = game.step(action)
            agent.train_short_memory(state, action, reward, next_state, done)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if render_mode:
                events = pygame.event.get() if pygame.display.get_init() else []
                for event in events:
                    if event.type == pygame.QUIT: pygame.quit(); quit()
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_p: game.pause()

        agent.train_long_memory()
        agent.update_epsilon()
        scores.append(score)

        full_save_path = os.path.join(MODELS_DIR, save_filename)
        if score > best_score:
            best_score = score
            agent.save(full_save_path, best_score)
            print(f"*** New best score! Model saved to {full_save_path} ***")
        
        last_state_path = os.path.join(MODELS_DIR, "last_model_state.pth")
        agent.save(last_state_path, best_score)
        
        print(f"Episode {episode} | Score: {score} | Best: {best_score} | Epsilon: {agent.epsilon:.3f} | Model: {save_filename}")
        
        if logger:
            log_data = {
                "Episode": episode, "Score": score, "Best Score": best_score, "Epsilon": agent.epsilon,
                "Total Reward": total_reward, "Steps": game.frame_iteration, "Model Type": model_type,
                "Model Name": os.path.basename(save_filename)
            }
            logger.log_episode(log_data)

    print(f"\nAverage score over {max_episodes} episodes: {np.mean(scores):.2f}")
    print(f"Final model state is in: {os.path.join(MODELS_DIR, save_filename)}")

    if logger:
        logger.save()