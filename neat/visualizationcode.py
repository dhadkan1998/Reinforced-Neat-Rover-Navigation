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

# these global variables to track fitness statistics
max_fitness_per_generation = []
avg_fitness_per_generation = []

green_zone_visit_info = {}

#track the number of cars that complete the job
starting_point_rewards_per_generation = []
four_green_zone_rewards_per_generation = []

# This will be reset every generation
visited_cells_this_generation = set()

class Car:
    green_count = 0
    starting_point_rewards = 0  # Track cars receiving the starting point reward
    four_green_zone_rewards = 0  # New counter for cars visiting 4 green zones     #temporary

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
        self.green_zone_entries = []  # List to record green zone entry coordinates
        self.visited_green_zones = set()  # Tracks grid cells of visited green zones
        self.stuck_check_interval = 100  # Time steps to check for stuck condition
        self.stuck_distance_threshold = 50  # Minimum distance to move to not be considered stuck
        self.position_history = []  # Track positions for stuck detection
        self.in_green_zone = False  # Tracks whether the car is in a green zone
        self.starting_point = (100, 100)
        self.track_record = []  # To store position history
        self.returning_home = False  # Flag to indicate returning to start
        self.visited_at_least_one_green_zone = False  # check if car visited greenzone for starting reward
        self.visited_green_zone_count = 0  # Add this line to track the number of visited green zones      #temporary
        
    def draw(self, screen):
        screen.blit(self.rotate_surface, self.pos)
        self.draw_radar(screen)

    def draw_radar(self, screen):
        for r in self.radars:
            pos, dist = r
            pygame.draw.line(screen, (0, 255, 0), self.center, pos, 1)
            pygame.draw.circle(screen, (0, 255, 0), pos, 5)

    def check_collision(self, map):
        # Assuming 'self.four_points' contains the points at the corners of the car
        # and the front of the car is defined by the first two points
        self.should_reverse = False
        front_points = self.four_points[:2]  # Assuming these are the front points
    
        # Introduce an offset to adjust the collision detection points
        offset = 40  # Adjust as needed based on the size of your car
        num_points = 5  # Increase the number of points for more accurate collision detection
        offset_points = []
        for i in range(num_points):
            # Calculate the point along the front of the car with an offset
            point_x = front_points[0][0] + i * (front_points[1][0] - front_points[0][0]) / (num_points - 1) + \
                      math.cos(math.radians(360 - self.angle)) * offset
            point_y = front_points[0][1] + i * (front_points[1][1] - front_points[0][1]) / (num_points - 1) + \
                      math.sin(math.radians(360 - self.angle)) * offset
            offset_points.append((point_x, point_y))
    
        for p in offset_points:
            # Check if the calculated point is within the bounds of the map image
            if 0 <= int(p[0]) < map.get_width() and 0 <= int(p[1]) < map.get_height():
                if map.get_at((int(p[0]), int(p[1]))) == (255, 255, 255, 255):  # Check if offset points hit a white obstacle
                    self.should_reverse = True
                    break
            else:
                # Handle the case where the point falls outside the map boundaries
                # Treat it as a collision
                self.should_reverse = True
                break
            
    def check_back_collision(self, map):
        back_x, back_y = self.get_back_points()  # Get the back detection point

        # Check if the back point collides with a white obstacle
        if map.get_at((back_x, back_y)) == (255, 255, 255, 255):  # Assuming white is the color of obstacles
            return True  # Collision detected
        return False  # No collision
    
    def update_stuck_status(self, map):
        # Check if the car is in a green zone
        x, y = int(self.center[0]), int(self.center[1])
        if map.get_at((x, y)) == (0, 128, 0, 255):  # Assuming (0, 255, 0, 255) is the RGBA color for green zones
            self.in_green_zone = True
        else:
            self.in_green_zone = False

        if not self.in_green_zone:
            # Add the current position to the history only if not in a green zone
            self.position_history.append(self.center.copy())
            # Only check if we have enough data and the car is not in a green zone
            if len(self.position_history) > self.stuck_check_interval:
                # Calculate the distance moved over the interval
                start_pos = self.position_history[-self.stuck_check_interval]
                end_pos = self.position_history[-1]
                distance_moved = math.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
                # Check if the car is stuck
                if distance_moved < self.stuck_distance_threshold:
                    self.is_alive = False  # Deactivate the rover
                # Remove the oldest position to keep the list size constant
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
    

#    def get_front_points(self):
#        # Calculate points at the front of the car
#        # Adjust the offset and angle as needed to position the detection point accurately at the car's front
#        offset = 40  # Distance from the center to the front of the car, adjust based on car size
#        angle_rad = math.radians(self.angle)
#
#        front_center_x = int(self.center[0] + math.cos(angle_rad) * offset)
#        front_center_y = int(self.center[1] + math.sin(angle_rad) * offset)
#
#        return (front_center_x, front_center_y)

    def get_back_points(self):
        # Calculate points at the back of the car
        # The offset should be adjusted based on the car size and how far back you want to detect
        offset = 40  # Distance from the center to the back of the car, adjust based on car size
        angle_rad = math.radians(self.angle)

        # Calculate the center back point
        back_center_x = int(self.center[0] - math.cos(angle_rad) * offset)
        back_center_y = int(self.center[1] - math.sin(angle_rad) * offset)

        # For wider coverage, you might also want to calculate the corners at the back
        # This example just calculates the center back point
        return (back_center_x, back_center_y)

    def search_for_green(self, map, generation_start_time):
        global green_zone_visit_info  # Ensure you're accessing the global variable

        x, y = int(self.center[0]), int(self.center[1])
        if map.get_at((x, y)) == (0, 128, 0, 255):
            grid_x = x // grid_cell_width
            grid_y = y // grid_cell_height
            current_green_zone = (grid_x, grid_y)

            if current_green_zone not in self.visited_green_zones:
                self.visited_green_zones.add(current_green_zone)
                self.visited_green_zone_count += 1  # Increment the count for each unique green zone visited    #temporary

                all_visited_green_zones.add(current_green_zone)  # Update global set

                self.goal = True
                Car.green_count += 1
                self.reached_green_time = pygame.time.get_ticks()
                self.should_reverse = True

            # Record the visit if it's the first time this zone is visited
            if current_green_zone not in green_zone_visit_info:
                current_time = pygame.time.get_ticks()
                time_taken = current_time - generation_start_time  # Assume generation_start_time is accessible
                green_zone_visit_info[current_green_zone] = (generation, time_taken)

            # Existing logic to handle first-time visitation
            self.visited_at_least_one_green_zone = True            

    def reverse(self, map):
        if self.should_reverse:
            if self.check_back_collision(map):  # Check for back collision
                self.speed = 8  # Stop the car if there's a collision at the back
                self.should_reverse = False  # Optionally, reset the reversing flag
                return  # Exit the method early

            # Continue with the reversing logic if no back collision is detected
            current_time = pygame.time.get_ticks()
            if self.reverse_start_time is None:
                self.reverse_start_time = current_time
                self.angle += 180  # Optionally, adjust the angle for better reversing behavior

            # Implement a time-based reversal, or until it's clear of obstacles
            if current_time - self.reverse_start_time < 1500:  # Reverse for 1.5 seconds, adjust as needed
                self.speed = -2  # Adjust the speed for effective reversing
            else:
                # Reset conditions to stop reversing
                self.should_reverse = False
                self.reverse_start_time = None
                self.speed = 0  # Optionally, adjust to resume forward motion or to stop


    def check_starting_position_reward(self):
        if self.visited_at_least_one_green_zone:
            starting_area_radius = 50  # Define a radius around the starting point that counts as "reaching" it
            distance_to_start = math.sqrt((self.pos[0] - self.starting_point[0])**2 + (self.pos[1] - self.starting_point[1])**2)
            if distance_to_start <= starting_area_radius:
                self.is_alive = False  # Or any other mechanism to make the car disappear or mark as completed
                Car.starting_point_rewards += 1   # Increase the fitness significantly as a reward
                if self.visited_green_zone_count >=3:
                    Car.four_green_zone_rewards += 1  # Increment the new counter   #temporary
                return True  # Indicate that the reward has been collected
        return False

    def update(self, map, generation_start_time):
        global visited_cells_this_generation
        current_time = pygame.time.get_ticks()
        if self.reached_green_time and current_time - self.reached_green_time < 5000: # If less than 5 seconds since car reached green
            self.speed = 0 
            return  # Do not update car's position

        self.reached_green_time = None 
        self.speed = 8
        self.reverse(map)  # Call the reverse method
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

        self.search_for_green(map, generation_start_time)

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
    # Draw vertical lines
    for x in range(0, map_width + 1, grid_cell_width):
        pygame.draw.line(screen, color, (x, 0), (x, map_height), thickness)
    # Draw horizontal lines
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
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Arial", 50)
    font = pygame.font.SysFont("Arial", 30)
    map = pygame.image.load('map11.png')

    generation_duration = 1
    generation_start_time = pygame.time.get_ticks()

    while True:
        current_time = pygame.time.get_ticks()
        elapsed_time = current_time - generation_start_time

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
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

                car.update(map, generation_start_time)
                car.update_stuck_status(map)  # Check if the car is stuck and update its status

                if car.check_starting_position_reward():
                    genomes[i][1].fitness += 100  # Adjust the reward magnitude as needed

                # Calculate the grid cell the car is currently in
                current_grid_cell = (int(car.center[0] // grid_cell_width), int(car.center[1] // grid_cell_height))
                
                if current_grid_cell not in visited_cells_this_generation:
                    visited_cells_this_generation.add(current_grid_cell)
                    # Reward for visiting a new cell. Adjust the reward magnitude as needed.
                    genomes[i][1].fitness += 1
        
        
                genomes[i][1].fitness += car.get_reward(map)

                if car.goal:
                    genomes[i][1].fitness += 10

                car_location = (int(car.center[0]), int(car.center[1]))

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

            # No need to reset all_visited_green_zones here
            
            generation += 1
            generation_start_time = pygame.time.get_ticks()

            Car.green_count = 0

            generation_numbers.append(generation)
            generation_fitness.append(max(g[1].fitness for g in genomes))

            # Record the green zone hits and fitness for this generation
            fitness_scores_per_generation.append(max(g[1].fitness for g in genomes))

            # Calculate maximum and average fitness for this generation
            max_fitness = max(g[1].fitness for g in genomes)
            avg_fitness = sum(g[1].fitness for g in genomes) / len(genomes)

            # Append the fitness statistics to the respective lists
            max_fitness_per_generation.append(max_fitness)
            avg_fitness_per_generation.append(avg_fitness)

            # Print or log the fitness statistics
            print(f"Generation: {generation}, Max Fitness: {max_fitness}, Average Fitness: {avg_fitness}")


            new_count = len(all_visited_green_zones)
            if new_count > unique_green_zones_count:
                unique_green_zones_count = new_count
                generations_without_increase = 0
                fixed_zone_flag = False  # Reset counter because we've seen an increase
            else:
                generations_without_increase += 1  # Increment counter
            
            # Append rewards data
            starting_point_rewards_per_generation.append(Car.starting_point_rewards)
            four_green_zone_rewards_per_generation.append(Car.four_green_zone_rewards)

            # Reset rewards for the next generation
            Car.starting_point_rewards = 0
            Car.four_green_zone_rewards = 0


            # Check if we've hit the threshold
            if generations_without_increase >= threshold_generations and not fixed_zone_flag:
                fixed_zone_flag = True  # Set the flag to display the message
            break


        screen.blit(map, (0, 0))
        for car in cars:
            if car.get_alive():
                car.draw(screen)

        # Inside your main game loop, where drawing happens
        font = pygame.font.SysFont("Arial", 30)
        unique_zones_text = font.render(f"Unique Green Zones Visited: {len(all_visited_green_zones)}", True, (255, 255, 0))
        screen.blit(unique_zones_text, (10, 10))  # Adjust position as needed

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
            fixed_zone_text_rect.center = (screen_width / 2, screen_height / 2)  # Adjust as needed
            screen.blit(fixed_zone_text, fixed_zone_text_rect)


        text = font.render("Remain Cars: " + str(remain_cars), True, (255, 255, 0))
        text_rect = text.get_rect()
        text_rect.center = (screen_width/2, 200)
        screen.blit(text, text_rect)
        
        job_complete_text = font.render(f"Cars Completed Job: {Car.starting_point_rewards}", True, (255, 255, 0))
        screen.blit(job_complete_text, (10, screen_height - 50))  # Adjust the position as needed

        # Inside your main game loop, after updating car states and before pygame.display.flip()
        #temporary
        four_zones_complete_text = font.render(f"Cars Completed 4 Green Zones & Returned: {Car.four_green_zone_rewards}", True, (255, 255, 0))
        screen.blit(four_zones_complete_text, (500, screen_height - 60))  # Adjust the position as needed


        draw_grid(screen, grid_cell_width, grid_cell_height, screen_width, screen_height)

        pygame.display.flip()
        clock.tick(0)

def save_green_zone_visits_to_file(filename="green_zone_visits.txt"):
    with open(filename, "w") as file:
        for zone, (gen, time_taken) in green_zone_visit_info.items():
            file.write(f"Zone: {zone}, First Visited: Generation {gen}, Time Taken: {time_taken} ms\n")

def save_rewards_to_file(filename="generation_rewards.txt"):
    with open(filename, "w") as file:
        file.write("Generation\tStarting Point Rewards\tFour Green Zone Rewards\n")
        for gen, spr, fgzr in zip(generation_numbers, starting_point_rewards_per_generation, four_green_zone_rewards_per_generation):
            file.write(f"{gen}\t\t\t\t\t\t\t\t{spr}\t\t\t\t\t\t\t\t\t{fgzr}\n")


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

    p.run(run_car, 20)

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
    original_map = Image.open('map11.png')
    blended_image = Image.blend(original_map.convert('RGBA'), heatmap_image.convert('RGBA'), alpha=0.5)
    
    # Save or show the blended image
    blended_image.save("colored_blended_map.png")
    blended_image.show()

    save_green_zone_visits_to_file()
    save_rewards_to_file()

    # Write max fitness and average fitness to a text file
    with open("fitness_statistics.txt", "w") as f:
        f.write("Generation\t\t\tMax Fitness\t\t\t\tAverage Fitness\n")
        for gen, max_fit, avg_fit in zip(generation_numbers, max_fitness_per_generation, avg_fitness_per_generation):
            f.write(f"{gen}\t\t\t\t\t\t{max_fit}\t\t\t\t\t\t{avg_fit}\n")

    # Print the statistics after the simulation ends
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

    plt.plot(generation_numbers, generation_fitness, marker='o')
    plt.title('Generation Number vs. Fitness')
    plt.savefig('GenFitness.png')
    
    plt.xlabel('Generation Number')
    plt.ylabel('Fitness')
    plt.show()