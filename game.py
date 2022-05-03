#importing all libraries need for the project

import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple

pygame.init()
font = pygame.font.SysFont('arial', 25)

# we need a reset function, after each game our agent should reset  the game and start fresh
# then the rewards our agent gets
#then a play function that computes the direction play(action) -> direction
#then     game iteration      function
#then a    is_collision       function

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
green = (124,252,0)
yellow = (255,255,0)
BLACK = (0,0,0)

#Snake attributes
BLOCK_SIZE = 20
SPEED = 400

class SnakeGameAI:

    def __init__(self, Width=640, Height=480):

        self.Width = Width
        self.Height = Height
        # init display
        self.display = pygame.display.set_mode((self.Width, self.Height))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        #calling the reset function
        self.reset()


    def reset(self):
        # init game state
        self.direction = Direction.RIGHT
        self.head = Point(self.Width/2, self.Height/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        #additionally we need to keep track of game or frame iteratiion
        #this is to keep track of score initialize with the zero
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    #Helper function to place the food 
    def _place_food(self):
        x = random.randint(0, (self.Width-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.Height-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    #here we use action instead of direction 
    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0
        game_over = False
        #if the snake doesnt improve for long time or improve
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food: #if the head hits the food then reward
            self.score += 1
            reward = 10
            #if it gets the food then it places a block
            self._place_food()
        else:
            #else we remove the last part
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score


    def is_collision(self, point=None):
        if point is None:
            point = self.head
        # hits boundary
        if point.x > self.Width - BLOCK_SIZE or point.x < 0 or point.y > self.Height - BLOCK_SIZE or point.y < 0:
            return True
        # hits itself
        if point in self.snake[1:]:
            return True

        return False


    def _update_ui(self):
        self.display.fill(BLACK)

        for point in self.snake:
            pygame.draw.rect(self.display, yellow, pygame.Rect(point.x, point.y, BLOCK_SIZE, BLOCK_SIZE))
            

        pygame.draw.rect(self.display, green, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render(" Points: " + str(self.score), True, yellow)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        current_indx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[current_indx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (current_indx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (current_indx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)