import os
from snake_game import SnakeGame
from dqn_agent import DQNAgent
import numpy as np
import pygame

def train_snake(render_mode=False, model_path=None, max_episodes=100):
    game = SnakeGame(w=20, h=20, block_size=25, render_mode=render_mode)
    state_size = len(game._get_state())
    action_size = 3  # [straight, right, left]
    agent = DQNAgent(state_size, action_size)
    scores = []
    best_score = 0

    # Możliwość wczytania modelu
    if model_path:
        if os.path.exists(model_path):
            best_score = agent.load(model_path)
            print(f"Wczytano model z: {model_path}")
        else:
            print(f"Nie znaleziono pliku {model_path}, zaczynam od zera.")

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
            # Obsługa pauzy (tylko w trybie graficznym)
            if render_mode:
                for event in game.display.get_events() if hasattr(game.display, 'get_events') else []:
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                        game.pause()
        agent.train_long_memory()
        agent.update_epsilon()
        scores.append(score)
        if score > best_score:
            best_score = score
            agent.save("best_model.pth", best_score)
        agent.save("last_model.pth", best_score)
        print(f"Episode {episode} | Score: {score} | Best: {best_score} | Epsilon: {agent.epsilon:.3f}")
    print(f"Average score over {max_episodes} episodes: {np.mean(scores):.2f}")

if __name__ == "__main__":
    # Prosty wybór trybu przez konsolę (na potrzeby testów)
    mode = input("Wybierz tryb: 1 - InRealTime, 2 - InBackground: ").strip()
    model_path = input("Podaj ścieżkę do pliku modelu (enter = nowy model): ").strip()
    train_snake(render_mode=(mode=="1"), model_path=(model_path if model_path else None), max_episodes=100) 