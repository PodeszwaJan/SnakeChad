import pygame
import sys
import os
import torch
from train import train_snake
import tkinter as tk
from tkinter import simpledialog

# --- Define paths relative to the project root ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
ASSETS_DIR = os.path.join(PROJECT_ROOT, 'assets')

# Initialize tkinter root for dialogs and hide it
root = tk.Tk()
root.withdraw()

pygame.init()

WIDTH, HEIGHT = 600, 500
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 128, 255)
FONT = pygame.font.SysFont('arial', 32)
SMALL_FONT = pygame.font.SysFont('arial', 24)
MUSIC_FILE = os.path.join(ASSETS_DIR, 'beethoven.mp3')

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Snake RL - Menu')

menu_items = [
    'Start InRealTime (Visible)',
    'Start InBackground (Fast Training)',
    'Select Existing Model',
    'Create and Train a New Model',
    'Set Number of Episodes',
    'Select Model Type (DQN/DDQN/ADQN)', # <-- MODIFIED
    'Exit'
]

selected = 0
model_to_load = None
save_filename = "new_model.pth"
max_episodes = 100
model_type = "DQN"
models = []
models_scores = []
selected_model_idx = 0
model_types = ["DQN", "DDQN", "ADQN"] # <-- MODIFIED

def refresh_models():
    """Scans the models directory for .pth files and their scores."""
    global models, models_scores
    models.clear()
    models_scores.clear()
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    for fname in os.listdir(MODELS_DIR):
        if fname.endswith('.pth'):
            try:
                checkpoint = torch.load(os.path.join(MODELS_DIR, fname))
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

    if model_to_load:
        model_info_text = f'Loading from: {os.path.basename(model_to_load)}'
    else:
        model_info_text = f'Saving new model as: {save_filename}'

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

    current_idx = model_types.index(model_type) # <-- MODIFIED
    
    for i, mtype in enumerate(model_types): # <-- MODIFIED
        color = BLUE if i == current_idx else WHITE
        text = SMALL_FONT.render(mtype, True, color)
        screen.blit(text, (WIDTH//2 - text.get_width()//2, 150 + i*40))
        
    pygame.display.flip()

def get_new_model_name_input():
    """Opens a dialog to get the filename for a new model."""
    try:
        name = simpledialog.askstring("New Model Name", "Enter a name for the new model file:", parent=root)
        if name:
            if not name.endswith('.pth'):
                name += '.pth'
            return name
    except Exception:
        pass
    return None

def menu_loop():
    """Main loop for handling menu navigation and actions."""
    global selected, model_to_load, max_episodes, selected_model_idx, model_type, save_filename
    
    refresh_models()

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
                    if selected == 0: run_game(render_mode=True)
                    elif selected == 1: run_game(render_mode=False)
                    elif selected == 2: choose_model()
                    elif selected == 3:
                        new_name = get_new_model_name_input()
                        if new_name:
                            model_to_load = None
                            save_filename = new_name
                    elif selected == 4: max_episodes = get_episodes_input()
                    elif selected == 5: choose_model_type()
                    elif selected == 6:
                        pygame.quit()
                        sys.exit()

def choose_model():
    """Loop for the model selection screen."""
    global selected_model_idx, model_to_load, save_filename
    refresh_models()
    choosing = True
    while choosing:
        draw_models_menu()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: selected_model_idx = (selected_model_idx - 1) % max(1, len(models))
                elif event.key == pygame.K_DOWN: selected_model_idx = (selected_model_idx + 1) % max(1, len(models))
                elif event.key == pygame.K_RETURN:
                    if models:
                        chosen_model = models[selected_model_idx]
                        model_to_load = chosen_model
                        save_filename = chosen_model
                    choosing = False
                elif event.key == pygame.K_ESCAPE: choosing = False

def choose_model_type():
    """Loop for the model type selection screen."""
    global model_type
    choosing = True
    current_idx = model_types.index(model_type)
    while choosing:
        draw_model_type_menu()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    current_idx = (current_idx - 1) % len(model_types)
                elif event.key == pygame.K_DOWN:
                    current_idx = (current_idx + 1) % len(model_types)
                elif event.key == pygame.K_RETURN or event.key == pygame.K_ESCAPE:
                    choosing = False
                model_type = model_types[current_idx]

def get_episodes_input():
    """Opens a dialog box to get the number of episodes from the user."""
    try:
        value = simpledialog.askinteger('Episodes', 'Enter the number of episodes:', minvalue=1, maxvalue=100000, parent=root)
        return value if value else 100
    except Exception:
        return 100

def run_game(render_mode):
    """Starts the training process and handles pre/post-training setup."""
    global model_to_load, max_episodes, model_type, save_filename
    if render_mode:
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(MUSIC_FILE)
            pygame.mixer.music.play(-1)
        except Exception as e:
            print(f'Could not play music: {e}')
    
    pygame.display.iconify()
    train_snake(render_mode=render_mode, model_path=model_to_load, max_episodes=max_episodes, model_type=model_type, save_filename=save_filename)
    
    if render_mode: pygame.mixer.music.stop()
    
    pygame.display.set_mode((WIDTH, HEIGHT))
    refresh_models()

if __name__ == '__main__':
    try:
        menu_loop()
    finally:
        root.destroy()
        pygame.quit()
        sys.exit()