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
MUSIC_FILE = 'beethoven.mp3'  # Upewnij się, że plik jest w katalogu

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Snake RL - Menu')

menu_items = [
    'Start InRealTime (widoczne ruchy)',
    'Start InBackground (szybki trening)',
    'Wybierz model',
    'Utwórz nowy model',
    'Ustaw liczbę epizodów',
    'Wyjście'
]

selected = 0
model_path = None
max_episodes = 100
models = []
models_scores = []
selected_model_idx = 0

# Wyszukaj dostępne modele i ich best_score
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
    screen.fill(BLACK)
    title = FONT.render('SNAKE RL - MENU', True, YELLOW)
    screen.blit(title, (WIDTH//2 - title.get_width()//2, 30))
    for i, item in enumerate(menu_items):
        color = BLUE if i == selected else WHITE
        text = SMALL_FONT.render(item, True, color)
        screen.blit(text, (WIDTH//2 - text.get_width()//2, 100 + i*40))
    # Info o modelu
    if model_path:
        info = SMALL_FONT.render(f'Model: {model_path}', True, YELLOW)
        screen.blit(info, (10, HEIGHT-60))
    if max_episodes:
        info2 = SMALL_FONT.render(f'Epizodów: {max_episodes}', True, YELLOW)
        screen.blit(info2, (10, HEIGHT-30))
    pygame.display.flip()

def draw_models():
    screen.fill(BLACK)
    title = FONT.render('Wybierz model', True, YELLOW)
    screen.blit(title, (WIDTH//2 - title.get_width()//2, 30))
    if not models:
        info = SMALL_FONT.render('Brak dostępnych modeli.', True, WHITE)
        screen.blit(info, (WIDTH//2 - info.get_width()//2, 150))
    for i, (fname, score) in enumerate(zip(models, models_scores)):
        color = BLUE if i == selected_model_idx else WHITE
        text = SMALL_FONT.render(f'{fname} (best_score: {score})', True, color)
        screen.blit(text, (WIDTH//2 - text.get_width()//2, 120 + i*35))
    pygame.display.flip()

def menu_loop():
    global selected, model_path, max_episodes, selected_model_idx
    while True:
        draw_menu()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected = (selected - 1) % len(menu_items)
                if event.key == pygame.K_DOWN:
                    selected = (selected + 1) % len(menu_items)
                if event.key == pygame.K_RETURN:
                    if selected == 0:  # InRealTime
                        run_game(render_mode=True)
                    elif selected == 1:  # InBackground
                        run_game(render_mode=False)
                    elif selected == 2:  # Wybierz model
                        choose_model()
                    elif selected == 3:  # Nowy model
                        model_path = None
                    elif selected == 4:  # Liczba epizodów
                        max_episodes = get_episodes()
                    elif selected == 5:  # Wyjście
                        pygame.quit()
                        sys.exit()

def choose_model():
    global selected_model_idx, model_path
    choosing = True
    while choosing:
        draw_models()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected_model_idx = (selected_model_idx - 1) % max(1, len(models))
                if event.key == pygame.K_DOWN:
                    selected_model_idx = (selected_model_idx + 1) % max(1, len(models))
                if event.key == pygame.K_RETURN:
                    if models:
                        model_path = models[selected_model_idx]
                    choosing = False
                if event.key == pygame.K_ESCAPE:
                    choosing = False

def get_episodes():
    import tkinter as tk
    from tkinter import simpledialog
    root = tk.Tk()
    root.withdraw()
    try:
        value = simpledialog.askinteger('Epizody', 'Podaj liczbę epizodów:', minvalue=1, maxvalue=100000)
        return value if value else 100
    except Exception:
        return 100
    finally:
        root.destroy()

def run_game(render_mode):
    global model_path, max_episodes
    if render_mode:
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(MUSIC_FILE)
            pygame.mixer.music.play(-1)  # zapętl
        except Exception as e:
            print(f'Nie można odtworzyć muzyki: {e}')
    pygame.display.iconify()  # Minimalizuj okno menu na czas treningu
    train_snake(render_mode=render_mode, model_path=model_path, max_episodes=max_episodes)
    if render_mode:
        pygame.mixer.music.stop()
    pygame.display.set_mode((WIDTH, HEIGHT))  # Przywróć menu

if __name__ == '__main__':
    menu_loop() 