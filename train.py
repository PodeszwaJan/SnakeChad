import os
from snake_game import SnakeGame
from dqn_agent import DQNAgent
from excel_logger import ExcelLogger
import numpy as np
import pygame

def train_snake(render_mode=False, model_path=None, max_episodes=100):
    game = SnakeGame(w=20, h=20, block_size=25, render_mode=render_mode)
    state_size = len(game._get_state())
    action_size = 3  # [straight, right, left]
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
            print(f"Loaded model from: {model_path}")
        else:
            print(f"Model file not found at {model_path}, starting from scratch.")

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
            
            # Handle pause (only in graphical mode)
            if render_mode:
                # Use a safe way to get events
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

        if score > best_score:
            best_score = score
            agent.save("best_model.pth", best_score)
        agent.save("last_model.pth", best_score)
        
        print(f"Episode {episode} | Score: {score} | Best Score: {best_score} | Epsilon: {agent.epsilon:.3f}")
        
        # Log to Excel if in background mode
        if logger:
            log_data = {
                "Episode": episode,
                "Score": score,
                "Best Score": best_score,
                "Epsilon": agent.epsilon,
                "Total Reward": total_reward,
                "Steps": game.frame_iteration
            }
            logger.log_episode(log_data)

    print(f"\nAverage score over {max_episodes} episodes: {np.mean(scores):.2f}")

    # Save the Excel file at the end of training
    if logger:
        logger.save()

if __name__ == "__main__":
    # Simple console mode selection (for testing purposes)
    mode = input("Select mode: 1 - InRealTime, 2 - InBackground: ").strip()
    model_path_input = input("Enter path to model file (press enter for a new model): ").strip()
    train_snake(render_mode=(mode=="1"), model_path=(model_path_input if model_path_input else None), max_episodes=100)