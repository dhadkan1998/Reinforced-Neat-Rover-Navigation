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
generation_max_fitness = []
generation_avg_fitness = []

all_three_zones_visited_per_generation = []
returned_after_three_zones_per_generation = []

screen_width = 1500
screen_height = 800
generation = 0

unique_green_zones_count = 0
generations_without_increase = 0
threshold_generations = 5  # You can adjust this threshold
fixed_zone_flag = False

# Define the size of each grid cell (e.g., matching the smallest green zone size)
grid_cell_width = 150
grid_cell_height = 150

# Initialize a set to track visited green zone cells
visited_green_cells = set()

# Initialize the car_location_record dictionary to keep track of car locations and visit counts
car_location_record = {}
car_position_frequency = {}
green_zone_hits_per_generation = []
fitness_scores_per_generation = []
all_green_zone_entries = []
all_visited_green_zones = set()

# These global variables to track fitness statistics
max_fitness_per_generation = []
avg_fitness_per_generation = []

green_zone_visit_info = {}

# Track the number of cars that complete the job
starting_point_rewards_per_generation = []
three_green_zone_rewards_per_generation = []

# This will be reset every generation
visited_cells_this_generation = set()

tracked_cars = {i: {
    "green_zone_order": [], 
    "green_zone_times": {}, 
    "visit_sequences": {}, 
    "all_zones_visited_generations": set(),
    "returning_home": False
} for i in range(50)}

class Car:
    green_count = 0
    starting_point_rewards = 0  # Track cars receiving the starting point reward
    three_green_zone_rewards = 0  # New counter for cars visiting 3 green zones
    all_three_zones_visited = 0  # New counter for cars visiting all 3 zones
    returned_after_three_zones = 0  # New counter for cars returning after visiting all 3 zones

    def __init__(self):
        self.surface = pygame.image.load("car.png")
        self.surface = pygame.transform.scale(self.surface, (80, 80))
        self.rotate_surface = self.surface
        self.pos = [100, 100]
        self.angle = 0
        self.speed = 0
        self.center = [self.pos[0] + 40, self.pos[1] + 40]
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
        self.all_zones_visited_generations = set()
        self.green_zone_entries = []  # List to record green zone entry coordinates
        self.visited_green_zones = set()  # Tracks grid cells of visited green zones
        self.stuck_check_interval = 100  # Time steps to check for stuck condition
        self.stuck_distance_threshold = 50  # Minimum distance to move to not be considered stuck
        self.position_history = []  # Track positions for stuck detection
        self.in_green_zone = False  # Tracks whether the car is in a green zone
        self.starting_point = (100, 100)
        self.track_record = []  # To store position history
        self.returning_home = False  # Flag to indicate returning to start
        self.visited_at_least_one_green_zone = False  # Check if car visited greenzone for starting reward
        self.visited_green_zone_count = 0  # Track the number of visited green zones
        self.green_zone_visit_order = []
        self.green_zone_visit_times = {}
        self.generation_start_time = None
        self.all_zones_visited_generation = None

    def draw(self, screen):
        screen.blit(self.rotate_surface, self.pos)
        self.draw_radar(screen)

    def draw_radar(self, screen):
        for r in self.radars:
            pos, dist = r
            pygame.draw.line(screen, (0, 255, 0), self.center, pos, 1)
            pygame.draw.circle(screen, (0, 255, 0), pos, 5)

    def set_generation_start_time(self, start_time):
        self.generation_start_time = start_time

    def record_green_zone_visit(self, zone, current_time):
        if zone not in self.green_zone_visit_order:
            self.green_zone_visit_order.append(zone)
            time_taken = current_time - self.generation_start_time
            self.green_zone_visit_times[zone] = time_taken

    def check_collision(self, map):
        self.should_reverse = False
        front_points = self.four_points[:2]
    
        offset = 40
        num_points = 5
        offset_points = []
        for i in range(num_points):
            point_x = front_points[0][0] + i * (front_points[1][0] - front_points[0][0]) / (num_points - 1) + \
                      math.cos(math.radians(360 - self.angle)) * offset
            point_y = front_points[0][1] + i * (front_points[1][1] - front_points[0][1]) / (num_points - 1) + \
                      math.sin(math.radians(360 - self.angle)) * offset
            offset_points.append((point_x, point_y))
    
        for p in offset_points:
            if 0 <= int(p[0]) < map.get_width() and 0 <= int(p[1]) < map.get_height():
                if map.get_at((int(p[0]), int(p[1]))) == (255, 255, 255, 255):
                    self.should_reverse = True
                    break
            else:
                self.should_reverse = True
                break
            
    def check_back_collision(self, map):
        back_x, back_y = self.get_back_points()
        if map.get_at((back_x, back_y)) == (255, 255, 255, 255):
            return True
        return False
    
    def update_stuck_status(self, map):
        x, y = int(self.center[0]), int(self.center[1])
        if map.get_at((x, y)) == (0, 128, 0, 255):
            self.in_green_zone = True
        else:
            self.in_green_zone = False

        if not self.in_green_zone:
            self.position_history.append(self.center.copy())
            if len(self.position_history) > self.stuck_check_interval:
                start_pos = self.position_history[-self.stuck_check_interval]
                end_pos = self.position_history[-1]
                distance_moved = math.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
                if distance_moved < self.stuck_distance_threshold:
                    self.is_alive = False
                self.position_history.pop(0)

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

    def get_back_points(self):
        offset = 40
        angle_rad = math.radians(self.angle)

        back_center_x = int(self.center[0] - math.cos(angle_rad) * offset)
        back_center_y = int(self.center[1] - math.sin(angle_rad) * offset)

        return (back_center_x, back_center_y)

    def search_for_green(self, map, generation_start_time):
        global green_zone_visit_info

        x, y = int(self.center[0]), int(self.center[1])
        if map.get_at((x, y)) == (0, 128, 0, 255):
            grid_x = x // grid_cell_width
            grid_y = y // grid_cell_height
            current_green_zone = (grid_x, grid_y)

            if current_green_zone not in self.visited_green_zones:
                self.visited_green_zones.add(current_green_zone)
                self.visited_green_zone_count += 1

                current_time = pygame.time.get_ticks()
                self.record_green_zone_visit(current_green_zone, current_time)
                all_visited_green_zones.add(current_green_zone)

                self.goal = True
                Car.green_count += 1
                self.reached_green_time = pygame.time.get_ticks()
                self.should_reverse = True

            if current_green_zone not in green_zone_visit_info:
                current_time = pygame.time.get_ticks()
                time_taken = current_time - generation_start_time
                green_zone_visit_info[current_green_zone] = (generation, time_taken)

            self.visited_at_least_one_green_zone = True            

    def reverse(self, map):
        if self.should_reverse:
            if self.check_back_collision(map):
                self.speed = 8
                self.should_reverse = False
                return

            current_time = pygame.time.get_ticks()
            if self.reverse_start_time is None:
                self.reverse_start_time = current_time
                self.angle += 180

            if current_time - self.reverse_start_time < 1500:
                self.speed = -2
            else:
                self.should_reverse = False
                self.reverse_start_time = None
                self.speed = 0

    def check_starting_position_reward(self):
        if self.visited_at_least_one_green_zone:
            starting_area_radius = 50
            distance_to_start = math.sqrt((self.pos[0] - self.starting_point[0])**2 + (self.pos[1] - self.starting_point[1])**2)
            if distance_to_start <= starting_area_radius:
                if hasattr(self, 'visited_all_three') and self.visited_all_three:
                    self.returning_home = True
                    Car.returned_after_three_zones += 1
                Car.starting_point_rewards += 1
                self.is_alive = False  # Mark car as completed its task
                return True
        return False


    def update(self, map, generation_start_time):
        global visited_cells_this_generation
        current_time = pygame.time.get_ticks()
        if self.reached_green_time and current_time - self.reached_green_time < 5000:
            self.speed = 0 
            return

        self.reached_green_time = None 
        self.speed = 8
        self.reverse(map)
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

        self.center = [int(self.pos[0]) + 40, int(self.pos[1]) + 40]
        len = 30
        left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * len, self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * len]
        right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * len, self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * len]
        left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * len, self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * len]
        right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * len, self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * len]
        self.four_points = [left_top, right_top, left_bottom, right_bottom]

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

        self.search_for_green(map, generation_start_time)

        if self.visited_green_zone_count == 3 and not hasattr(self, 'visited_all_three'):
            self.visited_all_three = True
            Car.all_three_zones_visited += 1

    def get_data(self):
        radars = self.radars
        ret = [0, 0, 0, 0, 0]
        for i, r in enumerate(radars):
            ret[i] = int(r[1] / 30)

        return ret

    def get_alive(self):
        return self.is_alive

    def get_reward(self, map):
        wall_penalty = 0
        for r in self.radars:
            _, dist = r
            if dist < 10:
                wall_penalty -= 2
    
        revisitation_penalty = -50 if self.check_revisitation() else 0
           
        return wall_penalty + revisitation_penalty

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


def draw_grid(screen, grid_cell_width, grid_cell_height, map_width, map_height, color=(255, 255, 255), thickness=1):
    for x in range(0, map_width + 1, grid_cell_width):
        pygame.draw.line(screen, color, (x, 0), (x, map_height), thickness)
    for y in range(0, map_height + 1, grid_cell_height):
        pygame.draw.line(screen, color, (0, y), (map_width, y), thickness)


def run_car(genomes, config):
    nets = []
    cars = []

    for id, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0
        cars.append(Car())

    pygame.init()
    map = pygame.image.load('map12.png')
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Arial", 50)
    font = pygame.font.SysFont("Arial", 30)

    generation_duration = 40000
    generation_start_time = pygame.time.get_ticks()
    for car in cars:
        car.set_generation_start_time(generation_start_time)

    while True:
        current_time = pygame.time.get_ticks()
        elapsed_time = current_time - generation_start_time

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
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

                car.update(map, generation_start_time)
                car.update_stuck_status(map)

                if car.check_starting_position_reward():
                    genomes[i][1].fitness += 80

                current_grid_cell = (int(car.center[0] // grid_cell_width), int(car.center[1] // grid_cell_height))
                
                if current_grid_cell not in visited_cells_this_generation:
                    visited_cells_this_generation.add(current_grid_cell)
                    genomes[i][1].fitness += 1
        
                genomes[i][1].fitness += car.get_reward(map)

                if car.goal:
                    genomes[i][1].fitness += 20

                if len(car.prev_positions) > 5:
                    car.prev_positions.pop(0)

        if remain_cars == 0 or elapsed_time >= generation_duration:
            global generation
            global unique_green_zones_count, generations_without_increase, fixed_zone_flag

            green_zone_hits_per_generation.append(Car.green_count)
            Car.green_count = 0
            all_green_zone_entries.clear()
            visited_green_cells.clear()
            visited_cells_this_generation.clear()

            generation += 1
            generation_start_time = pygame.time.get_ticks()

            generation_numbers.append(generation)
            generation_max_fitness.append(max(g[1].fitness for g in genomes))

            fitness_scores_per_generation.append(max(g[1].fitness for g in genomes))

            max_fitness = max(g[1].fitness for g in genomes)
            avg_fitness = sum(g[1].fitness for g in genomes) / len(genomes)

            max_fitness_per_generation.append(max_fitness)
            avg_fitness_per_generation.append(avg_fitness)

            new_count = len(all_visited_green_zones)
            if new_count > unique_green_zones_count:
                unique_green_zones_count = new_count
                generations_without_increase = 0
                fixed_zone_flag = False
            else:
                generations_without_increase += 1
            
            all_three_zones_visited_per_generation.append(Car.all_three_zones_visited)
            returned_after_three_zones_per_generation.append(Car.returned_after_three_zones)

            Car.starting_point_rewards = 0
            Car.three_green_zone_rewards = 0
            Car.all_three_zones_visited = 0
            Car.returned_after_three_zones = 0
            
            if generations_without_increase >= threshold_generations and not fixed_zone_flag:
                fixed_zone_flag = True

            update_tracked_cars(cars, generation)
            break

        screen.blit(map, (0, 0))
        for car in cars:
            if car.get_alive():
                car.draw(screen)

        unique_zones_text = font.render(f"Unique Green Zones Visited: {len(all_visited_green_zones)}", True, (255, 255, 0))
        screen.blit(unique_zones_text, (10, 10))

        green_count_text = font.render("Cars Reached Green: " + str(Car.green_count), True, (255, 255, 0))
        green_count_rect = green_count_text.get_rect()
        green_count_rect.center = (screen_width / 2, 250)
        screen.blit(green_count_text, green_count_rect)

        text = generation_font.render("Generation : " + str(generation), True, (255, 255, 0))
        text_rect = text.get_rect()
        text_rect.center = (screen_width/2, 100)
        screen.blit(text, text_rect)

        if fixed_zone_flag:
            fixed_zone_text = font.render("Room has been fixed", True, (255, 0, 0))
            fixed_zone_text_rect = fixed_zone_text.get_rect()
            fixed_zone_text_rect.center = (screen_width / 2, screen_height / 2)
            screen.blit(fixed_zone_text, fixed_zone_text_rect)


        text = font.render("Remain Cars: " + str(remain_cars), True, (255, 255, 0))
        text_rect = text.get_rect()
        text_rect.center = (screen_width/2, 200)
        screen.blit(text, text_rect)
        
        job_complete_text = font.render(f"Cars Completed Job: {Car.starting_point_rewards}", True, (255, 255, 0))
        screen.blit(job_complete_text, (10, screen_height - 50))

        three_zones_complete_text = font.render(f"Cars Completed 3 Green Zones & Returned: {Car.three_green_zone_rewards}", True, (255, 255, 0))
        screen.blit(three_zones_complete_text, (500, screen_height - 60))

        draw_grid(screen, grid_cell_width, grid_cell_height, screen_width, screen_height)

        pygame.display.flip()
        clock.tick(0)


def save_rewards_to_file(filename="generation_rewards.txt"):
    with open(filename, "w") as file:
        file.write("Generation\tNumber of Cars Visited All Three Zones\tNumber of Cars Returned After Visiting All Three Zones\n")
        for gen in generation_numbers:
            valid_cars_visited_all_zones = 0
            returned_cars = 0
            
            for car in tracked_cars.values():
                if gen in car["all_zones_visited_generations"]:
                    valid_cars_visited_all_zones += 1
                    if car["returning_home"]:
                        returned_cars += 1
            
            file.write(f"{gen}\t\t\t{valid_cars_visited_all_zones}\t\t\t{returned_cars}\n")

def update_tracked_cars(cars, generation):
    global tracked_cars
    for i, car in enumerate(cars[:50]):  # Only consider the first 50 cars
        current_gen_visits = []
        visited_zones_in_gen = set()

        for zone, time in car.green_zone_visit_times.items():
            if zone not in tracked_cars[i]["green_zone_times"]:
                tracked_cars[i]["green_zone_order"].append(zone)
                tracked_cars[i]["green_zone_times"][zone] = (time, generation)

            current_gen_visits.append((zone, time))
            visited_zones_in_gen.add(zone)

        tracked_cars[i]["visit_sequences"][generation] = current_gen_visits

        # Check if all 3 zones were visited this generation
        if len(visited_zones_in_gen) == 3:
            tracked_cars[i]["all_zones_visited_generations"].add(generation)
            print(f"Car {i} visited all 3 zones in Generation {generation}")  # Debug print statement

        # Initialize returning_home key if it doesn't exist
        if "returning_home" not in tracked_cars[i]:
            tracked_cars[i]["returning_home"] = False
        
        # Update returning_home status
        if car.check_starting_position_reward():
            tracked_cars[i]["returning_home"] = True

def save_car_data(filename="car_data.txt"):
    with open(filename, "w") as file:
        for i in range(50):
            file.write(f"Car: {i}\n")
            file.write(f"Green Zone Visit Order: {tracked_cars[i]['green_zone_order']}\n")
            file.write("Green Zone Visit Times:\n")
            for zone in tracked_cars[i]['green_zone_order']:
                time, gen = tracked_cars[i]['green_zone_times'][zone]
                file.write(f"Zone {zone}: {time} ms    Generation: {gen}\n")
            file.write("Visit Sequences Per Generation:\n")
            for gen, visits in sorted(tracked_cars[i]['visit_sequences'].items()):
                file.write(f"Generation {gen}: {visits}\n")
            if tracked_cars[i]["all_zones_visited_generations"]:
                file.write(f"Generations where all 3 zones were visited: {sorted(tracked_cars[i]['all_zones_visited_generations'])}\n")
                file.write("Times for each zone in the first generation where all zones were visited:\n")
                
                first_all_zones_gen = min(tracked_cars[i]['all_zones_visited_generations'])
                if first_all_zones_gen in tracked_cars[i]['visit_sequences']:
                    for zone, time in tracked_cars[i]['visit_sequences'][first_all_zones_gen]:
                        file.write(f"Zone {zone}: {time} ms\n")
                else:
                    print(f"Warning: Missing data for all 3 zones in Generation {first_all_zones_gen} for Car {i}")
                    file.write("Data for all zones in the specified generation is missing.\n")
            file.write("\n")  # Blank line between cars

if __name__ == "__main__":
    config_path = "./config-feedforward.txt"
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

    p.run(run_car, 1000)
    save_car_data()

    # Generating heatmap data
    heatmap_data = np.zeros((screen_height // 10, screen_width // 10))
    for (x, y), count in car_position_frequency.items():
        heatmap_data[y, x] = count

    # Apply Gaussian blur for smoothing
    smoothed_heatmap = gaussian_filter(heatmap_data, sigma=2)
    
    # Normalize the data for the full range of the colormap
    normalized_heatmap = smoothed_heatmap / np.max(smoothed_heatmap)
    
    # Choose a colormap
    colormap = plt.cm.get_cmap('hot')
    
    # Apply the colormap
    colored_heatmap = colormap(normalized_heatmap)
    
    # Convert to an image, drop the alpha channel and resize
    heatmap_image = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)
    heatmap_image = Image.fromarray(heatmap_image)
    heatmap_image = heatmap_image.resize((1500, 800), Image.BILINEAR)
    
    # Load the original map and blend it with the heatmap
    original_map = Image.open('map12.png')
    blended_image = Image.blend(original_map.convert('RGBA'), heatmap_image.convert('RGBA'), alpha=0.5)
    
    # Save or show the blended image
    blended_image.save("colored_blended_map.png")
    blended_image.show()

    save_rewards_to_file()

    with open("fitness_statistics.txt", "w") as f:
        f.write("Generation\t\t\tMax Fitness from that generation\t\t\t\tAverage Fitness from that generation\n")
        for gen, max_fit, avg_fit in zip(generation_numbers, max_fitness_per_generation, avg_fitness_per_generation):
            f.write(f"{gen}\t\t\t\t\t\t{max_fit}\t\t\t\t\t\t{avg_fit}\n")

    highest_green_zone_hits = max(green_zone_hits_per_generation)
    average_green_zone_hits = sum(green_zone_hits_per_generation) / len(green_zone_hits_per_generation)
    highest_fitness = max(fitness_scores_per_generation)
    average_fitness = sum(fitness_scores_per_generation) / len(fitness_scores_per_generation)

    print(f"Highest Number of Cars Reaching Green Zone: {highest_green_zone_hits}")
    print(f"Average Number of Cars Reaching Green Zone: {average_green_zone_hits}")
    print(f"Highest Fitness Score: {highest_fitness}")
    print(f"Average Fitness Score: {average_fitness}")

    with open("population_state.pkl", "wb") as f:
        pickle.dump(p, f)

    plt.figure(figsize=(10, 5))
    plt.plot(generation_numbers, max_fitness_per_generation, marker='o', label='Max Fitness')
    plt.title('Generation vs. Maximum Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Maximum Fitness')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('MaxFitness_vs_Generation.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(generation_numbers, avg_fitness_per_generation, marker='o', color='r', label='Average Fitness')
    plt.title('Generation vs. Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Average Fitness')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('AvgFitness_vs_Generation.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(generation_numbers, max_fitness_per_generation, marker='o', label='Max Fitness')
    plt.plot(generation_numbers, avg_fitness_per_generation, marker='o', color='r', label='Average Fitness')
    plt.title('Generation vs. Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('MaxAvgFitness_vs_Generation.png')

    plt.show()
