# Snake RL – Reinforcement Learning Snake Game

## Opis projektu

Projekt polega na trenowaniu agentów sztucznej inteligencji do gry w Snake przy użyciu różnych algorytmów uczenia ze wzmocnieniem (Reinforcement Learning, RL). Celem jest porównanie skuteczności i charakterystyki uczenia różnych podejść, takich jak DQN, DDQN, ADQN oraz PPO. Projekt oferuje graficzny interfejs użytkownika, możliwość wyboru i zapisu modeli oraz automatyczne logowanie wyników do plików Excel.

---

## Funkcjonalności

- **Wybór i trening różnych modeli RL:** DQN, DDQN, ADQN, PPO
- **Menu graficzne (Pygame):** Intuicyjna obsługa, wybór trybu, modelu, liczby epizodów, zarządzanie modelami
- **Trening z wizualizacją lub w tle:** Możliwość szybkiego treningu bez renderowania gry
- **Automatyczne logowanie wyników:** Wyniki każdego epizodu zapisywane do plików Excel
- **Łatwe wznawianie i testowanie modeli:** Wybór istniejących modeli i kontynuacja treningu

---

## Obsługiwane modele

- **DQN (Deep Q-Network):** Klasyczny algorytm RL z pojedynczą siecią Q
- **DDQN (Double DQN):** Dwie sieci Q, stabilniejsze uczenie, redukcja przeszacowania wartości Q
- **ADQN (Averaged DQN):** Uśrednianie wag kilku ostatnich modeli, jeszcze większa stabilność
- **PPO (Proximal Policy Optimization):** Metoda policy gradient z architekturą actor-critic, odporna na niestabilności

---

## Jak używać

1. Zainstaluj wymagane biblioteki (patrz sekcja Zależności)
2. Uruchom `python main.py`
3. Użyj strzałek ↑↓ do nawigacji w menu
4. Naciśnij Enter, aby wybrać opcję
5. Wybierz typ modelu (DQN/DDQN/ADQN/PPO)
6. Ustaw liczbę epizodów
7. Rozpocznij trening lub testowanie modelu

---

## Struktura plików

- `main.py` – Główne menu i interfejs użytkownika
- `train.py` – Logika treningu agentów
- `dqn_agent.py` – Implementacja DQN
- `ddqn_agent.py` – Implementacja DDQN
- `adqn_agent.py` – Implementacja ADQN
- `ppo_agent.py` – Implementacja PPO
- `snake_game.py` – Logika gry Snake i środowisko RL
- `excel_logger.py` – Logger zapisujący wyniki do plików Excel
- `requirements.txt` – Lista zależności
- `models/` – Zapisane modele
- `logs/` – Pliki z logami treningów

---

## Zależności

Wymagane biblioteki (możesz zainstalować poleceniem `pip install -r requirements.txt`):

```
torch
numpy
pygame
openpyxl
```

---

## Opis środowiska gry

Środowisko to klasyczna gra Snake zaimplementowana w Pygame. Stan gry jest reprezentowany jako wektor cech (pozycja węża, jedzenia, kierunek ruchu, zagrożenia). Agent podejmuje decyzje o ruchu (prosto, w lewo, w prawo), a środowisko zwraca nagrodę, nowy stan i informację o zakończeniu gry.

---

## Opis agentów

- **DQN:** Uczy się wartości Q dla każdej akcji w danym stanie, korzystając z pojedynczej sieci neuronowej i replay memory.
- **DDQN:** Wykorzystuje dwie sieci (główną i docelową) do stabilniejszego uczenia i redukcji przeszacowania Q.
- **ADQN:** Uśrednia wagi kilku ostatnich modeli, co dodatkowo stabilizuje proces uczenia.
- **PPO:** Metoda policy gradient z architekturą actor-critic, optymalizuje politykę w sposób odporny na duże zmiany.

---

## Logowanie i analiza wyników

- Wyniki każdego epizodu (score, best score, epsilon, reward, liczba kroków, typ modelu) są automatycznie zapisywane do plików Excel w katalogu `logs/`.
- Możesz analizować postępy treningu, porównywać modele i wizualizować wyniki (np. w Excelu lub Pandas/Matplotlib).

---

## Porównanie modeli

| Model | Zalety | Wady |
|-------|--------|------|
| DQN   | Prosty, szybki trening | Może przeszacowywać wartości Q |
| DDQN  | Stabilniejszy trening, lepsze wyniki | Wolniejszy trening, większe zużycie pamięci |
| ADQN  | Jeszcze większa stabilność, odporność na niestabilności | Większe wymagania obliczeniowe i pamięciowe |
| PPO   | Bardzo stabilny, odporny na niestabilności, nowoczesny | Wymaga więcej epizodów i strojenia hiperparametrów |

---

## Przykładowe uruchomienie

```bash
pip install -r requirements.txt
python main.py
```

---

- Implementacje bazują na klasycznych artykułach RL (DQN, DDQN, PPO) oraz dokumentacji PyTorch i Pygame.

---

