import pygame
import sys
import os
import torch
from train import train_snake

pygame.init()

WIDTH, HEIGHT = 600, 500
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 128, 255)
FONT = pygame.font.SysFont('arial', 32)
SMALL_FONT = pygame.font.SysFont('arial', 24)
MUSIC_FILE = 'beethoven.mp3'  # Make sure this file is in the directory

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Snake RL - Menu')

menu_items = [
    'Start InRealTime (Visible)',
    'Start InBackground (Fast Training)',
    'Select Existing Model',
    'Start With a New Model',
    'Set Number of Episodes',
    'Select Model Type (DQN/DDQN)',
    'Exit'
]

selected = 0
model_path = None
max_episodes = 100
model_type = "DQN"  # Default model type
models = []
models_scores = []
selected_model_idx = 0

def refresh_models():
    """Scans the directory for .pth model files and their scores."""
    global models, models_scores
    models.clear()
    models_scores.clear()
    for fname in os.listdir('.'):
        if fname.endswith('.pth'):
            try:
                checkpoint = torch.load(fname)
                score = checkpoint.get('best_score', 0)
                models.append(fname)
                models_scores.append(score)
            except Exception:
                pass

def draw_menu():
    """Draws the main menu screen."""
    screen.fill(BLACK)
    title = FONT.render('SNAKE RL - MENU', True, YELLOW)
    screen.blit(title, (WIDTH//2 - title.get_width()//2, 30))

    for i, item in enumerate(menu_items):
        color = BLUE if i == selected else WHITE
        text = SMALL_FONT.render(item, True, color)
        screen.blit(text, (WIDTH//2 - text.get_width()//2, 100 + i*40))

    # Display info about the selected model, episode count, and model type
    if model_path:
        model_info_text = f'Model: {os.path.basename(model_path)}'
    else:
        model_info_text = 'Model: (A new model will be created)'
    
    info = SMALL_FONT.render(model_info_text, True, YELLOW)
    screen.blit(info, (10, HEIGHT-90))
    
    info2 = SMALL_FONT.render(f'Episodes: {max_episodes}', True, YELLOW)
    screen.blit(info2, (10, HEIGHT-60))
    
    info3 = SMALL_FONT.render(f'Model Type: {model_type}', True, YELLOW)
    screen.blit(info3, (10, HEIGHT-30))
    
    pygame.display.flip()

def draw_models_menu():
    """Draws the model selection screen."""
    screen.fill(BLACK)
    title = FONT.render('Select Model', True, YELLOW)
    screen.blit(title, (WIDTH//2 - title.get_width()//2, 30))

    if not models:
        info = SMALL_FONT.render('No saved models found.', True, WHITE)
        screen.blit(info, (WIDTH//2 - info.get_width()//2, 150))
    
    for i, (fname, score) in enumerate(zip(models, models_scores)):
        color = BLUE if i == selected_model_idx else WHITE
        text_str = f'{fname} (Best Score: {score})'
        text = SMALL_FONT.render(text_str, True, color)
        screen.blit(text, (WIDTH//2 - text.get_width()//2, 120 + i*35))
        
    pygame.display.flip()

def draw_model_type_menu():
    """Draws the model type selection screen."""
    screen.fill(BLACK)
    title = FONT.render('Select Model Type', True, YELLOW)
    screen.blit(title, (WIDTH//2 - title.get_width()//2, 30))

    model_types = ["DQN", "DDQN"]
    
    for i, mtype in enumerate(model_types):
        color = BLUE if mtype == model_type else WHITE
        text = SMALL_FONT.render(mtype, True, color)
        screen.blit(text, (WIDTH//2 - text.get_width()//2, 150 + i*50))
        
    pygame.display.flip()

def menu_loop():
    """Main loop for handling menu navigation and actions."""
    global selected, model_path, max_episodes, selected_model_idx, model_type
    
    refresh_models() # Initial scan for models

    while True:
        draw_menu()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected = (selected - 1) % len(menu_items)
                elif event.key == pygame.K_DOWN:
                    selected = (selected + 1) % len(menu_items)
                elif event.key == pygame.K_RETURN:
                    if selected == 0:  # InRealTime
                        run_game(render_mode=True)
                    elif selected == 1:  # InBackground
                        run_game(render_mode=False)
                    elif selected == 2:  # Select Model
                        choose_model()
                    elif selected == 3:  # New Model
                        model_path = None # This now works as intended
                    elif selected == 4:  # Number of Episodes
                        max_episodes = get_episodes_input()
                    elif selected == 5:  # Model Type
                        choose_model_type()
                    elif selected == 6:  # Exit
                        pygame.quit()
                        sys.exit()

def choose_model():
    """Loop for the model selection screen."""
    global selected_model_idx, model_path
    refresh_models() # Refresh model list every time we enter this screen
    choosing = True
    while choosing:
        draw_models_menu()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected_model_idx = (selected_model_idx - 1) % max(1, len(models))
                elif event.key == pygame.K_DOWN:
                    selected_model_idx = (selected_model_idx + 1) % max(1, len(models))
                elif event.key == pygame.K_RETURN:
                    if models:
                        model_path = models[selected_model_idx]
                    choosing = False
                elif event.key == pygame.K_ESCAPE:
                    choosing = False

def choose_model_type():
    """Loop for the model type selection screen."""
    global model_type
    choosing = True
    while choosing:
        draw_model_type_menu()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    model_type = "DQN" if model_type == "DDQN" else "DDQN"
                elif event.key == pygame.K_DOWN:
                    model_type = "DQN" if model_type == "DDQN" else "DDQN"
                elif event.key == pygame.K_RETURN:
                    choosing = False
                elif event.key == pygame.K_ESCAPE:
                    choosing = False

def get_episodes_input():
    """Opens a dialog box to get the number of episodes from the user."""
    import tkinter as tk
    from tkinter import simpledialog
    root = tk.Tk()
    root.withdraw()
    try:
        value = simpledialog.askinteger('Episodes', 'Enter the number of episodes:', minvalue=1, maxvalue=100000)
        return value if value else 100
    except Exception:
        return 100
    finally:
        root.destroy()

def run_game(render_mode):
    """Starts the training process and handles pre/post-training setup."""
    global model_path, max_episodes, model_type
    if render_mode:
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(MUSIC_FILE)
            pygame.mixer.music.play(-1)  # Loop indefinitely
        except Exception as e:
            print(f'Could not play music: {e}')
    
    pygame.display.iconify()  # Minimize menu window during training
    train_snake(render_mode=render_mode, model_path=model_path, max_episodes=max_episodes, model_type=model_type)
    
    if render_mode:
        pygame.mixer.music.stop()
    
    pygame.display.set_mode((WIDTH, HEIGHT))  # Restore menu window
    refresh_models() # Refresh model list after training in case new ones were saved

if __name__ == '__main__':
    menu_loop()