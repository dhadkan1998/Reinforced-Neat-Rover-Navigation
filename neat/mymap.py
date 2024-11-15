import pygame
import os
import math
import sys
import random
import neat

screen_width = 1500
screen_height = 800
generation = 0

class Car:
    def __init__(self, *args, **kwargs):
        self.surface = pygame.image.load("car.png")
        self.surface = pygame.transform.scale(self.surface, (100, 100))
        self.rotate_surface = self.surface

        if 'initial_position' in kwargs:
            self.pos = kwargs['initial_position']
        else:
            self.pos = [random.randint(20, screen_width - 120), random.randint(20, screen_height - 120)]

        self.angle = 0
        self.speed = 0
        self.center = [self.pos[0] + 50, self.pos[1] + 50]
        self.radars = []
        self.radars_for_draw = []
        self.is_alive = True
        self.goal = False
        self.distance = 0
        self.time_spent = 0

    def draw(self, screen):
        screen.blit(self.rotate_surface, self.pos)
        self.draw_radar(screen)

    def draw_radar(self, screen):
        for r in self.radars:
            pos, dist = r
            pygame.draw.line(screen, (0, 255, 0), self.center, pos, 1)
            pygame.draw.circle(screen, (0, 255, 0), pos, 5)

    def check_collision(self, map):
        for p in self.four_points:
            if map.get_at((int(p[0]), int(p[1]))) == (255, 255, 255, 255):
                self.is_alive = False
                break

    def check_radar(self, degree, map):
        len = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * len)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * len)

        while x < map.get_width() and y < map.get_height() and not map.get_at((x, y)) == (255, 255, 255, 255) and len < 200:
            len += 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * len)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * len)

        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([(x, y), dist])

    def update(self, map):
        self.speed = 8

        self.rotate_surface = self.rot_center(self.surface, self.angle)
        self.pos[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        if self.pos[0] < 20:
            self.pos[0] = 20
        elif self.pos[0] > screen_width - 120:
            self.pos[0] = screen_width - 120

        self.distance += self.speed
        self.time_spent += 1
        self.pos[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        if self.pos[1] < 20:
            self.pos[1] = 20
        elif self.pos[1] > screen_height - 120:
            self.pos[1] = screen_height - 120

        self.center = [int(self.pos[0]) + 50, int(self.pos[1]) + 50]
        len = 40
        left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * len, self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * len]
        right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * len, self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * len]
        left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * len, self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * len]
        right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * len, self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * len]
        self.four_points = [left_top, right_top, left_bottom, right_bottom]

        self.check_collision(map)
        self.radars.clear()
        for d in range(-90, 120, 45):
            self.check_radar(d, map)

    def get_data(self):
        radars = self.radars
        ret = [0, 0, 0, 0, 0]
        for i, r in enumerate(radars):
            ret[i] = int(r[1] / 30)

        return ret

    def get_alive(self):
        return self.is_alive

    def get_reward(self):
        center_x, center_y = screen_width / 2, screen_height / 2
        distance_to_center = math.sqrt((self.center[0] - center_x)**2 + (self.center[1] - center_y)**2)

        center_reward_zone = 50
        outer_reward_zone = 100

        if distance_to_center < center_reward_zone:
            reward = 5
        elif distance_to_center < outer_reward_zone:
            reward = 2
        else:
            reward = 0

        return reward

    def rot_center(self, image, angle):
        orig_rect = image.get_rect()
        rot_image = pygame.transform.rotate(image, angle)
        rot_rect = orig_rect.copy()
        rot_rect.center = rot_image.get_rect().center
        rot_image = rot_image.subsurface(rot_rect).copy()
        return rot_image

def run_car(genomes, config):
    nets = []
    cars = []

    initial_x = random.randint(20, screen_width - 120)
    initial_y = random.randint(20, screen_height - 120)

    for id, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0
        cars.append(Car(initial_position=[initial_x, initial_y]))

    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Arial", 50)
    font = pygame.font.SysFont("Arial", 30)
    map_image = pygame.image.load('map1.png')

    generation_duration = 20000  # 1 minute in milliseconds
    generation_start_time = pygame.time.get_ticks()

    while True:
        current_time = pygame.time.get_ticks()
        elapsed_time = current_time - generation_start_time

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        for index, car in enumerate(cars):
            output = nets[index].activate(car.get_data())
            i = output.index(max(output))
            if i == 0:
                car.angle += 10
            else:
                car.angle -= 10

        remain_cars = 0
        max_reward_car = None
        max_reward_value = float('-inf')

        for i, car in enumerate(cars):
            if car.get_alive():
                remain_cars += 1
                car.update(map_image)
                genomes[i][1].fitness += car.get_reward()

            if genomes[i][1].fitness > max_reward_value:
                max_reward_value = genomes[i][1].fitness
                max_reward_car = car

        if remain_cars == 0 or elapsed_time >= generation_duration:
            # Proceed to the next generation
            global generation
            generation += 1

            # Print generation statistics
            print(f"Generation {generation} - Time: {elapsed_time}ms - Max Reward: {max_reward_value}")

            # Start the next generation
            generation_start_time = pygame.time.get_ticks()
            break

        screen.blit(map_image, (0, 0))
        for car in cars:
            if car.get_alive():
                car.draw(screen)

        text = generation_font.render("Generation : " + str(generation), True, (255, 255, 0))
        text_rect = text.get_rect()
        text_rect.center = (screen_width/2, 100)
        screen.blit(text, text_rect)

        text = font.render("remain cars : " + str(remain_cars), True, (255, 255, 0))
        text_rect = text.get_rect()
        text_rect.center = (screen_width/2, 200)
        screen.blit(text, text_rect)

        if max_reward_car:
            text_max_reward = font.render("Max Reward Car: " + str(max_reward_value), True, (255, 255, 0))
            text_max_reward_rect = text_max_reward.get_rect()
            text_max_reward_rect.center = (max_reward_car.center[0], max_reward_car.center[1] + 60)
            screen.blit(text_max_reward, text_max_reward_rect)

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    config_path = "./config-feedforward.txt"
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    p.run(run_car, 1000)
