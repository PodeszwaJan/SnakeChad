import os
import numpy as np
import pygame
from snake_game import SnakeGame
from dqn_agent import DQNAgent
from ddqn_agent import DDQNAgent
from adqn_agent import ADQNAgent
from ppo_agent import PPOAgent
from excel_logger import ExcelLogger

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# --- MODIFIED: PPO-specific constant for updating ---
# The agent will perform a learning update after this many episodes have been played.
PPO_UPDATE_EVERY_N_EPISODES = 4

def train_snake(render_mode=False, model_path=None, max_episodes=100, model_type="DQN", save_filename=None):
    game = SnakeGame(w=20, h=20, block_size=25, render_mode=render_mode)
    state_size = len(game._get_state())
    action_size = 3

    if save_filename is None:
        save_filename = f"{model_type.lower()}_model.pth"

    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    # --- Agent creation block ---
    if model_type == "PPO":
        agent = PPOAgent(state_size, action_size)
    elif model_type == "DDQN":
        agent = DDQNAgent(state_size, action_size)
    elif model_type == "ADQN":
        agent = ADQNAgent(state_size, action_size)
    else: # Default to DQN
        agent = DQNAgent(state_size, action_size)
    
    scores = []
    best_score = 0
    logger = ExcelLogger() if not render_mode else None
    
    # --- MODIFIED: Counter for PPO updates ---
    episode_counter_for_update = 0

    if model_path:
        full_model_path = os.path.join(MODELS_DIR, model_path)
        if os.path.exists(full_model_path):
            best_score = agent.load(full_model_path)
            print(f"Loaded {model_type} model from: {full_model_path} | Best score: {best_score}")
        else:
            print(f"Model file not found at {full_model_path}, starting from scratch.")
    else:
        print(f"Starting new training. Best model will be saved as '{save_filename}'.")

    # --- Main training loop ---
    for episode in range(1, max_episodes + 1):
        state = game.reset()
        done = False
        total_reward = 0
        current_score = 0
        
        # --- Episode gameplay loop ---
        while not done:
            # PPO and DQN agents select actions differently
            if model_type == "PPO":
                action_idx, log_prob, value = agent.get_action(state)
                # Convert action index to one-hot for the game environment
                action_one_hot = np.zeros(action_size)
                action_one_hot[action_idx] = 1
                next_state, reward, done, current_score = game.step(action_one_hot)
                # PPO remembers more info than DQN
                agent.remember(state, action_idx, log_prob, reward, done, value)
            
            else: # DQN, DDQN, ADQN training loop
                action_one_hot = agent.get_action(state)
                next_state, reward, done, current_score = game.step(action_one_hot)
                agent.train_short_memory(state, action_one_hot, reward, next_state, done)
                agent.remember(state, action_one_hot, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            # Handle pygame events (like closing window or pausing)
            if render_mode:
                events = pygame.event.get() if pygame.display.get_init() else []
                for event in events:
                    if event.type == pygame.QUIT: pygame.quit(); quit()
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_p: game.pause()
        
        # --- MODIFIED: Post-episode learning logic ---
        if model_type == "PPO":
            episode_counter_for_update += 1
            if episode_counter_for_update % PPO_UPDATE_EVERY_N_EPISODES == 0:
                print("\n--- Updating PPO Agent ---")
                agent.update()
                episode_counter_for_update = 0
        else:
            # DQN-style agents learn from long-term memory after every episode
            agent.train_long_memory()
            agent.update_epsilon()
        
        # --- Logging and Saving ---
        scores.append(current_score)
        full_save_path = os.path.join(MODELS_DIR, save_filename)
        
        if current_score > best_score:
            best_score = current_score
            agent.save(full_save_path, best_score)
            print(f"*** New best score! Model saved to {full_save_path} ***")
        
        epsilon_str = f"Epsilon: {agent.epsilon:.3f}" if hasattr(agent, 'epsilon') else "Epsilon: N/A (PPO)"
        print(f"Episode {episode} | Score: {current_score} | Best: {best_score} | {epsilon_str} | Model: {save_filename}")
        
        if logger:
            log_data = {
                "Episode": episode, "Score": current_score, "Best Score": best_score,
                "Epsilon": agent.epsilon if hasattr(agent, 'epsilon') else -1,
                "Total Reward": total_reward, "Steps": game.frame_iteration, "Model Type": model_type,
                "Model Name": os.path.basename(save_filename)
            }
            logger.log_episode(log_data)

    print(f"\nAverage score over {max_episodes} episodes: {np.mean(scores):.2f}")
    print(f"Final model state is in: {os.path.join(MODELS_DIR, save_filename)}")

    if logger:
        logger.save()