# Snake RL - Reinforcement Learning Snake Game

## Nowe funkcjonalności

### 1. Wybór modelu (DQN vs DDQN)
- **DQN (Deep Q-Network)**: Klasyczny algorytm uczenia ze wzmocnieniem
- **DDQN (Double Deep Q-Network)**: Ulepszona wersja DQN z dwoma sieciami neuronowymi

#### Różnice między DQN a DDQN:
- **DQN**: Używa jednej sieci do wyboru i oceny akcji
- **DDQN**: Używa głównej sieci do wyboru akcji i sieci docelowej do oceny, co redukuje nadmierną estymację wartości Q

### 2. Poprawiona logika zapisywania modeli
- Modele są teraz zapisywane do pliku, który został utworzony/załadowany
- DQN: `dqn_model.pth`
- DDQN: `ddqn_model.pth`
- Nie ma już automatycznego zapisywania do `best_model.pth` i `last_model.pth`

### 3. Menu główne
- **Start InRealTime (Visible)**: Trening z wizualizacją
- **Start InBackground (Fast Training)**: Szybki trening w tle
- **Select Existing Model**: Wybór istniejącego modelu
- **Start With a New Model**: Tworzenie nowego modelu
- **Set Number of Episodes**: Ustawienie liczby epizodów
- **Select Model Type (DQN/DDQN)**: Wybór typu modelu
- **Exit**: Wyjście z programu

## Jak używać

1. Uruchom `python main.py`
2. Użyj strzałek ↑↓ do nawigacji w menu
3. Naciśnij Enter aby wybrać opcję
4. Wybierz typ modelu (DQN lub DDQN)
5. Ustaw liczbę epizodów
6. Rozpocznij trening

## Pliki

- `main.py` - Główny plik z menu
- `train.py` - Logika treningu
- `dqn_agent.py` - Implementacja DQN
- `ddqn_agent.py` - Implementacja DDQN
- `snake_game.py` - Gra Snake
- `excel_logger.py` - Logger do Excel
- `requirements.txt` - Zależności

## Zależności

```
torch
pygame
numpy
openpyxl
```

## Porównanie modeli

| Model | Zalety | Wady |
|-------|--------|------|
| DQN | Prosty, szybki trening | Może przeszacowywać wartości Q |
| DDQN | Stabilniejszy trening, lepsze wyniki | Wolniejszy trening, więcej pamięci | 