import numpy as np
from scipy.spatial.distance import cdist

from config import CONFIG


class PSO:
    def __init__(self, params, hospital_coords, population_coords):
        self.params = params
        self.hospital_coords = hospital_coords
        self.population_coords = population_coords

        self.n_particles = CONFIG['N_PARTICLES']
        self.max_iterations = CONFIG['MAX_ITERATIONS']
        self.w = CONFIG['INERTIA_WEIGHT']
        self.c1 = CONFIG['COGNITIVE_PARAM']
        self.c2 = CONFIG['SOCIAL_PARAM']

        self.particles = np.random.uniform(
            low=[params['LAT_MIN'], params['LON_MIN']],
            high=[params['LAT_MAX'], params['LON_MAX']],
            size=(self.n_particles, 2)
        )

        self.velocities = np.random.uniform(
            low=-0.01,
            high=0.01,
            size=(self.n_particles, 2)
        )

        self.pbest_positions = self.particles.copy()
        self.pbest_fitness = np.array([self.evaluate(p) for p in self.particles])

        self.gbest_idx = np.argmax(self.pbest_fitness)
        self.gbest_position = self.pbest_positions[self.gbest_idx].copy()
        self.gbest_fitness = self.pbest_fitness[self.gbest_idx]

        self.fitness_history = [self.gbest_fitness]
        self.iteration_count = 0
        self.iteration_found = 0

    def evaluate(self, position):
        position = np.array(position).reshape(1, -1)

        distances_to_hospitals = cdist(position, self.hospital_coords)[0]
        min_distance_to_hospitals = np.min(distances_to_hospitals)

        distances_to_population = cdist(position, self.population_coords)[0]

        distances_sorted = np.sort(distances_to_population)
        k = min(10, len(distances_sorted))
        avg_k_nearest_deg = distances_sorted[:k].mean()
        avg_k_nearest_km = avg_k_nearest_deg * 111

        population_in_strict = np.sum(distances_to_population <= self.params['STRICT_DISTANCE'])
        population_in_coverage = np.sum(distances_to_population <= self.params['MIN_COVERAGE_DISTANCE'])

        fitness = 0.0

        if min_distance_to_hospitals >= self.params['MIN_DISTANCE_FROM_HOSPITAL']:
            fitness += 100.0
        else:
            penalty = (self.params['MIN_DISTANCE_FROM_HOSPITAL'] - min_distance_to_hospitals) * 500
            fitness -= penalty

        max_pop_km = CONFIG['MAX_POPULATION_DISTANCE_KM']
        if avg_k_nearest_km <= max_pop_km:
            fitness += 80.0
        else:
            penalty = (avg_k_nearest_km - max_pop_km) * 30.0
            fitness -= penalty

        if population_in_strict >= self.params['MIN_POPULATION_CENTERS_IN_RANGE']:
            fitness += 150.0
        else:
            penalty = (self.params['MIN_POPULATION_CENTERS_IN_RANGE'] - population_in_strict) * 30.0
            fitness -= penalty

        fitness += population_in_strict * 20.0
        fitness += population_in_coverage * 5.0

        avg_all_population_km = distances_to_population.mean() * 111
        fitness -= avg_all_population_km * 2.0

        fitness += min_distance_to_hospitals * 111 * 3.0

        return fitness

    def optimize(self):
        print("\nRUNNING PSO OPTIMIZATION (REVISED FITNESS)")
        print("=" * 110 + "\n")

        for iteration in range(self.max_iterations):
            self.iteration_count = iteration + 1

            for i in range(self.n_particles):
                r1 = np.random.rand(2)
                r2 = np.random.rand(2)

                self.velocities[i] = (
                    self.w * self.velocities[i]
                    + self.c1 * r1 * (self.pbest_positions[i] - self.particles[i])
                    + self.c2 * r2 * (self.gbest_position - self.particles[i])
                )

                self.velocities[i] = np.clip(self.velocities[i], -0.05, 0.05)

                self.particles[i] += self.velocities[i]

                self.particles[i] = np.clip(
                    self.particles[i],
                    [self.params['LAT_MIN'], self.params['LON_MIN']],
                    [self.params['LAT_MAX'], self.params['LON_MAX']]
                )

                fitness = self.evaluate(self.particles[i])

                if fitness > self.pbest_fitness[i]:
                    self.pbest_fitness[i] = fitness
                    self.pbest_positions[i] = self.particles[i].copy()

            best_idx = np.argmax(self.pbest_fitness)
            if self.pbest_fitness[best_idx] > self.gbest_fitness:
                self.gbest_fitness = self.pbest_fitness[best_idx]
                self.gbest_position = self.pbest_positions[best_idx].copy()
                self.iteration_found = self.iteration_count

            self.fitness_history.append(self.gbest_fitness)

            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1:3d}: gBest Fitness = {self.gbest_fitness:.2f}")

        return self.gbest_position, self.gbest_fitness
