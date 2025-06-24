import os
from snake_game import SnakeGame
from dqn_agent import DQNAgent
from ddqn_agent import DDQNAgent
from excel_logger import ExcelLogger
import numpy as np
import pygame

def train_snake(render_mode=False, model_path=None, max_episodes=100, model_type="DQN", save_filename=None):
    game = SnakeGame(w=20, h=20, block_size=25, render_mode=render_mode)
    state_size = len(game._get_state())
    action_size = 3  # [straight, right, left]
    
    # Set default save name if none is provided
    if save_filename is None:
        save_filename = "dqn_model.pth" if model_type == "DQN" else "ddqn_model.pth"

    # Create agent based on model type
    if model_type == "DDQN":
        agent = DDQNAgent(state_size, action_size)
    else:  # DQN
        agent = DQNAgent(state_size, action_size)
    
    scores = []
    best_score = 0
    logger = None

    # Initialize logger only for background training
    if not render_mode:
        logger = ExcelLogger()

    # Option to load a model
    if model_path:
        if os.path.exists(model_path):
            best_score = agent.load(model_path)
            print(f"Loaded {model_type} model from: {model_path} | Best score: {best_score}")
        else:
            print(f"Model file not found at {model_path}, starting from scratch.")
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
            
            # Handle game events
            if render_mode:
                events = pygame.event.get() if pygame.display.get_init() else []
                for event in events:
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                        game.pause()

        agent.train_long_memory()
        agent.update_epsilon()
        scores.append(score)

        # Save model with current best score
        if score > best_score:
            best_score = score
            agent.save(save_filename, best_score)
            print(f"*** New best score! Model saved to {save_filename} ***")
        
        # Always save the last state for continuity in a separate file
        agent.save("last_model_state.pth", best_score)
        
        print(f"Episode {episode} | Score: {score} | Best: {best_score} | Epsilon: {agent.epsilon:.3f} | Model: {os.path.basename(save_filename)}")
        
        # Log to Excel if in background mode
        if logger:
            log_data = {
                "Episode": episode,
                "Score": score,
                "Best Score": best_score,
                "Epsilon": agent.epsilon,
                "Total Reward": total_reward,
                "Steps": game.frame_iteration,
                "Model Type": model_type,
                "Model Name": os.path.basename(save_filename)
            }
            logger.log_episode(log_data)

    print(f"\nAverage score over {max_episodes} episodes: {np.mean(scores):.2f}")
    print(f"Final model state is in: {save_filename}")

    # Save the Excel file at the end of training
    if logger:
        logger.save()

if __name__ == "__main__":
    # This console mode is for quick testing, main.py is the primary entry point.
    mode = input("Select mode: 1 - InRealTime, 2 - InBackground: ").strip()
    model_type_input = input("Select model type (DQN/DDQN): ").strip().upper()
    if model_type_input not in ["DQN", "DDQN"]:
        model_type_input = "DQN"
    
    model_path_input = input("Enter path to model file to load (press enter for a new model): ").strip()
    
    if model_path_input:
        save_file = model_path_input
    else:
        save_file = input(f"Enter a name for the new {model_type_input} model file: ").strip()
        if not save_file:
            save_file = f"new_{model_type_input.lower()}_model.pth"

    train_snake(
        render_mode=(mode=="1"), 
        model_path=(model_path_input if model_path_input else None), 
        max_episodes=100,
        model_type=model_type_input,
        save_filename=save_file
    )