import numpy as np
import random
from collections import namedtuple
import pygame

Point = namedtuple('Point', 'x, y')

class Direction:
    RIGHT = 0
    LEFT = 1
    UP = 2
    DOWN = 3

class SnakeGame:
    def __init__(self, w=20, h=20, block_size=20, render_mode=False):
        self.w = w
        self.h = h
        self.block_size = block_size
        self.render_mode = render_mode
        if self.render_mode:
            pygame.init()
            self.display = pygame.display.set_mode((self.w * self.block_size, self.h * self.block_size))
            pygame.display.set_caption('Snake RL')
            self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [self.head,
                      Point(self.head.x - 1, self.head.y),
                      Point(self.head.x - 2, self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        if self.render_mode:
            self.render()
        return self._get_state()

    def _place_food(self):
        while True:
            x = random.randint(0, self.w - 1)
            y = random.randint(0, self.h - 1)
            self.food = Point(x, y)
            if self.food not in self.snake:
                break

    def _get_state(self):
        head = self.snake[0]
        point_l = Point(head.x - 1, head.y)
        point_r = Point(head.x + 1, head.y)
        point_u = Point(head.x, head.y - 1)
        point_d = Point(head.x, head.y + 1)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and self._is_collision(point_r)) or
            (dir_l and self._is_collision(point_l)) or
            (dir_u and self._is_collision(point_u)) or
            (dir_d and self._is_collision(point_d)),
            # Danger right
            (dir_u and self._is_collision(point_r)) or
            (dir_d and self._is_collision(point_l)) or
            (dir_l and self._is_collision(point_u)) or
            (dir_r and self._is_collision(point_d)),
            # Danger left
            (dir_d and self._is_collision(point_r)) or
            (dir_u and self._is_collision(point_l)) or
            (dir_r and self._is_collision(point_u)) or
            (dir_l and self._is_collision(point_d)),
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # Food location
            self.food.x < head.x,  # food left
            self.food.x > head.x,  # food right
            self.food.y < head.y,  # food up
            self.food.y > head.y   # food down
        ]
        return np.array(state, dtype=int)

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.snake[0]
        # Hits boundary
        if pt.x < 0 or pt.x >= self.w or pt.y < 0 or pt.y >= self.h:
            return True
        # Hits itself
        if pt in self.snake[1:]:
            return True
        return False

    def render(self):
        if not self.render_mode:
            return
        self.display.fill((0, 0, 0))
        # Draw snake
        for pt in self.snake:
            pygame.draw.rect(self.display, (0, 255, 0), pygame.Rect(pt.x * self.block_size, pt.y * self.block_size, self.block_size, self.block_size))
        # Draw food
        pygame.draw.rect(self.display, (255, 0, 0), pygame.Rect(self.food.x * self.block_size, self.food.y * self.block_size, self.block_size, self.block_size))
        # Draw score
        font = pygame.font.SysFont('arial', 25)
        text = font.render(f'Score: {self.score}', True, (255, 255, 255))
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        self.clock.tick(100)  # FPS
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

    def step(self, action):
        self.frame_iteration += 1
        # action: [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clock_wise[(idx + 1) % 4]  # right turn r->d->l->u
        else:  # [0, 0, 1]
            new_dir = clock_wise[(idx - 1) % 4]  # left turn
        self.direction = new_dir

        x = self.snake[0].x
        y = self.snake[0].y
        if self.direction == Direction.RIGHT:
            x += 1
        elif self.direction == Direction.LEFT:
            x -= 1
        elif self.direction == Direction.UP:
            y -= 1
        elif self.direction == Direction.DOWN:
            y += 1
        new_head = Point(x, y)

        reward = 0
        game_over = False
        if self._is_collision(new_head) or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return self._get_state(), reward, game_over, self.score

        self.snake.insert(0, new_head)
        if new_head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        if self.render_mode:
            self.render()
        return self._get_state(), reward, game_over, self.score

    def pause(self):
        if self.render_mode:
            paused = True
            font = pygame.font.SysFont('arial', 40)
            while paused:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_p:
                            paused = False
                self.display.blit(font.render('PAUZA (P aby wznowić)', True, (255,255,0)), [self.w * self.block_size // 4, self.h * self.block_size // 2])
                pygame.display.flip()
                self.clock.tick(5)

    def restart(self):
        return self.reset() 