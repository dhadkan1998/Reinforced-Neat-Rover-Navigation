import pygame
import os
import math
import sys
import pickle
import random
import neat
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image

generation_numbers = []
generation_fitness = []

screen_width = 1500
screen_height = 800
generation = 0

# Initialize the car_location_record dictionary to keep track of car locations and visit counts
car_location_record = {}
car_position_frequency = {}

tracked_car_index = 23

class Car:
    green_count = 0

    def __init__(self):
        self.surface = pygame.image.load("car.png")
        self.surface = pygame.transform.scale(self.surface, (100, 100))
        self.rotate_surface = self.surface
        self.pos = [100, 100]
        self.angle = 0
        self.speed = 0
        self.center = [self.pos[0] + 50, self.pos[1] + 50]
        self.radars = []
        self.radars_for_draw = []
        self.is_alive = True
        self.goal = False
        self.distance = 0
        self.time_spent = 0
        self.current_location = None
        self.prev_positions = []  # List to track previous positions
        self.pause_time = 0
        self.reached_green_time = None
        self.should_reverse = False
        self.reverse_start_time = None
        self.green_zone_entries = []  # List to record green zone entry coordinates
        self.cluster_count = 0  # Number of clusters entered by the car
        self.in_red_zone = False

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
            len = len + 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * len)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * len)

        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([(x, y), dist])

    def is_clicked(self, mouse_pos):
        car_rect = self.rotate_surface.get_rect(topleft=self.pos)
        return car_rect.collidepoint(mouse_pos)

    def search_for_green(self, map):
        x, y = int(self.center[0]), int(self.center[1])
        if map.get_at((x, y)) == (0, 128, 0, 255) and not self.goal:
            self.goal = True
            Car.green_count += 1
            self.reached_green_time = pygame.time.get_ticks()  # Record the time car reached the green zone
            self.should_reverse = True  # Set the reverse flag

            # Record green zone entry coordinates
            entry_coord = (x, y)
            self.green_zone_entries.append(entry_coord)

    def search_for_red(self, map):
        x, y = int(self.center[0]), int(self.center[1])
        if map.get_at((x, y)) == (255, 0, 0, 255):
            self.in_red_zone = True
        else:
            self.in_red_zone = False

    def reverse(self):
        if self.should_reverse:
            current_time = pygame.time.get_ticks()
            if self.reverse_start_time is None:
                self.reverse_start_time = current_time

            # Check if 3 seconds have passed since reversing started
            if current_time - self.reverse_start_time < 1500:
                self.speed = -8  # Reverse at a speed of -16 (opposite direction)
            else:
                self.should_reverse = False
                self.reverse_start_time = None
                self.speed = 0  # Stop reversing

    def update(self, map):
        current_time = pygame.time.get_ticks()
        if self.reached_green_time and current_time - self.reached_green_time < 5000: # If less than 5 seconds since car reached green
            self.speed = 0 
            return  # Do not update car's position

        self.reached_green_time = None 
        self.speed = 16
        self.reverse()  # Call the reverse method
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


        # Track the car's position (reducing resolution for heatmap)
        heatmap_x = int(self.center[0] / 10)
        heatmap_y = int(self.center[1] / 10)
        if (heatmap_x, heatmap_y) in car_position_frequency:
            car_position_frequency[(heatmap_x, heatmap_y)] += 1
        else:
            car_position_frequency[(heatmap_x, heatmap_y)] = 1

        self.check_collision(map)
        self.radars.clear()
        for d in range(-90, 120, 45):
            self.check_radar(d, map)

        self.search_for_green(map)

    def get_data(self):
        radars = self.radars
        ret = [0, 0, 0, 0, 0]
        for i, r in enumerate(radars):
            ret[i] = int(r[1] / 30)

        return ret

    def get_alive(self):
        return self.is_alive

    def get_reward(self, map, cluster_threshold):
        wall_penalty = 0
        for r in self.radars:
            _, dist = r
            if dist < 10:
                wall_penalty -= 2
    
        revisitation_penalty = -50 if self.check_revisitation() else 0
    
        red_zone_penalty = -50 if self.in_red_zone else 0
    
        # Cluster green zone entries
        clusters = dynamic_cluster(self.green_zone_entries, cluster_threshold)
        
        # Check if the car entered a new cluster
        if len(clusters) > self.cluster_count:
            self.cluster_count = len(clusters)
            return 10  # Reward for entering a new cluster
        elif len(clusters) == self.cluster_count and self.goal:
            return -5  # Penalty for entering a green zone in an existing cluster
    
        return red_zone_penalty + wall_penalty + revisitation_penalty
    

    def rot_center(self, image, angle):
        orig_rect = image.get_rect()
        rot_image = pygame.transform.rotate(image, angle)
        rot_rect = orig_rect.copy()
        rot_rect.center = rot_image.get_rect().center
        rot_image = rot_image.subsurface(rot_rect).copy()
        return rot_image

    def check_revisitation(self):
        current_location = (int(self.center[0]), int(self.center[1]))
        for prev_location in self.prev_positions:
            distance = math.sqrt((prev_location[0] - current_location[0])**2 + (prev_location[1] - current_location[1])**2)
            if distance <= 5:
                return True
        return False

def dynamic_cluster(entries, cluster_threshold):
    clusters = []
    for entry in entries:
        # Check if the entry belongs to an existing cluster
        added_to_cluster = False
        for cluster in clusters:
            for coord in cluster:
                distance = math.sqrt((coord[0] - entry[0]) ** 2 + (coord[1] - entry[1]) ** 2)
                if distance <= cluster_threshold:
                    cluster.append(entry)
                    added_to_cluster = True
                    break
            if added_to_cluster:
                break
        if not added_to_cluster:
            # Create a new cluster for this entry
            clusters.append([entry])
    return clusters

def run_car(genomes, config):
    nets = []
    cars = []

    for id, g in genomes:
        net = neat.nn.RecurrentNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0
        cars.append(Car())

    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Arial", 50)
    font = pygame.font.SysFont("Arial", 30)
    map = pygame.image.load('map3.png')

    generation_duration = 20000
    generation_start_time = pygame.time.get_ticks()
    cluster_threshold = 30  # Adjust this value for clustering sensitivity

    while True:
        current_time = pygame.time.get_ticks()
        elapsed_time = current_time - generation_start_time

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button clicked
                    mouse_pos = pygame.mouse.get_pos()
                    for car in cars:
                        if car.get_alive() and car.is_clicked(mouse_pos):
                            car.is_alive = False

        for index, car in enumerate(cars):
            output = nets[index].activate(car.get_data())
            i = output.index(max(output))
            if i == 0:
                car.angle += 10
            else:
                car.angle -= 10

        remain_cars = 0

        for i, car in enumerate(cars):
            if car.get_alive():
                remain_cars += 1

                car.update(map)
                genomes[i][1].fitness += car.get_reward(map, cluster_threshold)

                if car.goal:
                    genomes[i][1].fitness += 10

                car_location = (int(car.center[0]), int(car.center[1]))

                if len(car.prev_positions) > 5:
                    car.prev_positions.pop(0)

        if remain_cars == 0 or elapsed_time >= generation_duration:
            global generation
            generation += 1
            generation_start_time = pygame.time.get_ticks()

            Car.green_count = 0

            generation_numbers.append(generation)
            generation_fitness.append(max(g[1].fitness for g in genomes))

            break

        screen.blit(map, (0, 0))
        for car in cars:
            if car.get_alive():
                car.draw(screen)

        green_count_text = font.render("Cars Reached Green: " + str(Car.green_count), True, (255, 255, 0))
        green_count_rect = green_count_text.get_rect()
        green_count_rect.center = (screen_width / 2, 250)
        screen.blit(green_count_text, green_count_rect)

        text = generation_font.render("Generation : " + str(generation), True, (255, 255, 0))
        text_rect = text.get_rect()
        text_rect.center = (screen_width/2, 100)
        screen.blit(text, text_rect)

        text = font.render("Remain Cars: " + str(remain_cars), True, (255, 255, 0))
        text_rect = text.get_rect()
        text_rect.center = (screen_width/2, 200)
        screen.blit(text, text_rect)

        pygame.display.flip()
        clock.tick(0)

if __name__ == "__main__":
    config_path = "./rnn-config-feedforward.txt"
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    if os.path.exists("population_state.pkl"):
        with open("population_state.pkl", "rb") as f:
            p = pickle.load(f)
    else:
        p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    p.run(run_car, 1500)

    # Generating heatmap data
    heatmap_data = np.zeros((screen_height // 10, screen_width // 10))
    for (x, y), count in car_position_frequency.items():
        heatmap_data[y, x] = count

    # Apply Gaussian blur for smoothing
    smoothed_heatmap = gaussian_filter(heatmap_data, sigma=2)
    
    # Normalize the data for the full range of the colormap
    normalized_heatmap = smoothed_heatmap / np.max(smoothed_heatmap)
    
    # Choose a colormap
    colormap = plt.cm.get_cmap('hot')  # Replace 'hot' with your chosen colormap
    
    # Apply the colormap
    colored_heatmap = colormap(normalized_heatmap)
    
    # Convert to an image, drop the alpha channel and resize
    heatmap_image = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)
    heatmap_image = Image.fromarray(heatmap_image)
    heatmap_image = heatmap_image.resize((1500, 800), Image.BILINEAR)
    
    # Load the original map and blend it with the heatmap
    original_map = Image.open('map3.png')
    blended_image = Image.blend(original_map.convert('RGBA'), heatmap_image.convert('RGBA'), alpha=0.5)
    
    # Save or show the blended image
    blended_image.save("colored_blended_map.png")
    blended_image.show()

    with open("population_state.pkl", "wb") as f:
        pickle.dump(p, f)

    plt.plot(generation_numbers, generation_fitness, marker='o')
    plt.title('Generation Number vs. Fitness')
    plt.xlabel('Generation Number')
    plt.ylabel('Fitness')
    plt.show()
