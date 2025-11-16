import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import warnings

from config import CONFIG

warnings.filterwarnings('ignore')


def analyze_results(
    optimal_location,
    params,
    districts_df,
    hospitals_df,
    districts_coords,
    hospitals_coords,
    pso
):
    optimal_location = np.array(optimal_location).reshape(1, -1)

    distances_to_hospitals = cdist(optimal_location, hospitals_coords)[0]
    distances_to_population = cdist(optimal_location, districts_coords)[0]

    distances_sorted = np.sort(distances_to_population)
    k = min(10, len(distances_sorted))
    avg_k_nearest_km = distances_sorted[:k].mean() * 111

    population_in_strict = np.sum(distances_to_population <= params['STRICT_DISTANCE'])
    population_in_coverage = np.sum(distances_to_population <= params['MIN_COVERAGE_DISTANCE'])

    print("\n" + "=" * 110)
    print("OPTIMIZATION RESULTS (REVISED FITNESS)")
    print("=" * 110)

    print(f"\n Optimal Location for New Cardiac Hospital:")
    print(f"  Latitude:  {optimal_location[0, 0]:.6f}°")
    print(f"  Longitude: {optimal_location[0, 1]:.6f}°")
    print(f"  Fitness Score: {pso.gbest_fitness:.2f}")
    print(f"  Found at Iteration: {pso.iteration_found}/{pso.iteration_count}")

    print(f"\n Distance to Existing Hospitals:")
    for i, (_, row) in enumerate(hospitals_df.iterrows()):
        dist_km = distances_to_hospitals[i] * 111
        status = "✓" if dist_km >= CONFIG['MIN_DISTANCE_FROM_HOSPITAL_KM'] else "✗"
        print(f"  {status} {i+1:2d}. {row['Hospital Name']:.<45} {dist_km:>6.2f} km")

    print(f"\n Strict Constraint - Districts within {CONFIG['STRICT_DISTANCE_KM']:.1f} km:")
    pop_strict_list = []
    for i, (_, row) in enumerate(districts_df.iterrows()):
        if distances_to_population[i] <= params['STRICT_DISTANCE']:
            pop_strict_list.append((row['District Name'], distances_to_population[i]))

    print(f"  Found: {len(pop_strict_list)}/{CONFIG['MIN_POPULATION_CENTERS_IN_RANGE']} (REQUIRED)\n")
    for idx, (name, dist) in enumerate(sorted(pop_strict_list, key=lambda x: x[1]), 1):
        dist_km = dist * 111
        print(f"  {idx}. {name:.<45} {dist_km:>6.2f} km")

    print(f"\n Coverage - Districts within {CONFIG['MIN_COVERAGE_DISTANCE_KM']:.1f} km:")
    print(f"  Found: {population_in_coverage} districts\n")

    print(f" Performance Summary:")
    print(f"  Minimum Distance to Hospital: {distances_to_hospitals.min() * 111:.2f} km")
    print(f"  Average Distance to Hospitals: {distances_to_hospitals.mean() * 111:.2f} km")
    print(f"  Average Distance to Population (all): {distances_to_population.mean() * 111:.2f} km")
    print(f"  Average Distance to {k} nearest districts: {avg_k_nearest_km:.2f} km")

    return {
        'optimal_location': optimal_location[0],
        'distances_to_hospitals': distances_to_hospitals,
        'distances_to_population': distances_to_population,
        'population_in_strict': len(pop_strict_list),
        'population_in_coverage': int(population_in_coverage),
        'avg_k_nearest_km': float(avg_k_nearest_km),
        'k_nearest': k,
    }


def create_visualization(
    optimal_location,
    params,
    districts_df,
    hospitals_df,
    districts_coords,
    hospitals_coords,
    pso,
    analysis_results,
    output_path="pso_revised_fitness.png"
):
    optimal_location = np.array(optimal_location)
    distances_to_population = analysis_results['distances_to_population']

    fig = plt.figure(figsize=(24, 14))

    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(pso.fitness_history, 'b-', linewidth=2.5, marker='o', markersize=5)
    if pso.iteration_found > 0:
        ax1.axvline(
            x=pso.iteration_found,
            color='g', linestyle='--', linewidth=2.5,
            label=f'Best Found (Iter {pso.iteration_found})'
        )
    ax1.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('gBest Fitness Score', fontsize=12, fontweight='bold')
    ax1.set_title('PSO Convergence (Revised Fitness)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)

    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(
        hospitals_coords[:, 1], hospitals_coords[:, 0],
        c='red', s=350, marker='X', label='Existing Hospitals', zorder=5,
        edgecolors='darkred', linewidth=2
    )
    ax2.scatter(
        optimal_location[1], optimal_location[0],
        c='lime', s=700, marker='*', label='Optimal New Location', zorder=5,
        edgecolors='darkgreen', linewidth=3
    )

    circle_strict = plt.Circle(
        (optimal_location[1], optimal_location[0]),
        params['STRICT_DISTANCE'],
        color='red', fill=False, linestyle='-',
        linewidth=3, label=f"Strict ({CONFIG['STRICT_DISTANCE_KM']:.1f} km)"
    )
    ax2.add_patch(circle_strict)

    circle_coverage = plt.Circle(
        (optimal_location[1], optimal_location[0]),
        params['MIN_COVERAGE_DISTANCE'],
        color='green', fill=False, linestyle='--',
        linewidth=2.5, label=f"Coverage ({CONFIG['MIN_COVERAGE_DISTANCE_KM']:.1f} km)"
    )
    ax2.add_patch(circle_coverage)

    ax2.scatter(
        districts_coords[:, 1], districts_coords[:, 0],
        c='lightblue', s=180, marker='o', label='Crowded Districts',
        alpha=0.6, edgecolors='darkblue', linewidth=1.5
    )

    pop_strict = districts_coords[distances_to_population <= params['STRICT_DISTANCE']]
    if len(pop_strict) > 0:
        ax2.scatter(
            pop_strict[:, 1], pop_strict[:, 0],
            c='red', s=200, marker='s', edgecolors='darkred',
            linewidth=2, zorder=4,
            label=f"In Strict ({CONFIG['STRICT_DISTANCE_KM']:.1f} km)"
        )

    pop_coverage = districts_coords[
        (distances_to_population <= params['MIN_COVERAGE_DISTANCE']) &
        (distances_to_population > params['STRICT_DISTANCE'])
    ]
    if len(pop_coverage) > 0:
        ax2.scatter(
            pop_coverage[:, 1], pop_coverage[:, 0],
            c='gold', s=200, marker='o', edgecolors='orange',
            linewidth=2.5, zorder=4,
            label=f"In Coverage ({CONFIG['STRICT_DISTANCE_KM']:.1f}-{CONFIG['MIN_COVERAGE_DISTANCE_KM']:.1f} km)"
        )

    ax2.set_xlabel('Longitude (°)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Latitude (°)', fontsize=12, fontweight='bold')
    ax2.set_title('Optimal Hospital Location Map', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(2, 3, 3)
    hospital_names = [name.split()[0] for name in hospitals_df['Hospital Name']]
    distances_km = analysis_results['distances_to_hospitals'] * 111
    colors = ['green' if d >= CONFIG['MIN_DISTANCE_FROM_HOSPITAL_KM'] else 'red' for d in distances_km]

    ax3.barh(
        range(len(hospital_names)),
        distances_km,
        color=colors, alpha=0.75, edgecolor='black', linewidth=1.5
    )
    ax3.axvline(
        x=CONFIG['MIN_DISTANCE_FROM_HOSPITAL_KM'],
        color='r', linestyle='--', linewidth=2.5,
        label=f"Min ({CONFIG['MIN_DISTANCE_FROM_HOSPITAL_KM']:.1f} km)"
    )
    ax3.set_yticks(range(len(hospital_names)))
    ax3.set_yticklabels(hospital_names, fontsize=8)
    ax3.set_xlabel('Distance (km)', fontsize=12, fontweight='bold')
    ax3.set_title('Distance to Existing Hospitals', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.legend(fontsize=9)

    ax4 = plt.subplot(2, 3, 4)
    ax4.axis('off')
    config_text = f"""REVISED FITNESS FUNCTION


Min Distance from Hospital: {CONFIG['MIN_DISTANCE_FROM_HOSPITAL_KM']} km
Strict Distance: {CONFIG['STRICT_DISTANCE_KM']} km
Min Districts (Strict): {CONFIG['MIN_POPULATION_CENTERS_IN_RANGE']}
Coverage Distance: {CONFIG['MIN_COVERAGE_DISTANCE_KM']} km


PSO Parameters:
  Particles: {CONFIG['N_PARTICLES']}
  Iterations: {CONFIG['MAX_ITERATIONS']}
  Inertia: {CONFIG['INERTIA_WEIGHT']}
"""

    ax4.text(
        0.05, 0.95, config_text,
        fontsize=10, family='monospace',
        verticalalignment='top',
        bbox=dict(
            boxstyle='round',
            facecolor='lightyellow',
            alpha=0.8,
            pad=1.2
        ),
        fontweight='bold'
    )

    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    results_text = f"""SOLUTION (REVISED)


Latitude:  {optimal_location[0]:.6f}°
Longitude: {optimal_location[1]:.6f}°


Fitness: {pso.gbest_fitness:.2f}
Iteration: {pso.iteration_found}/{pso.iteration_count}


Min to Hospital: {analysis_results['distances_to_hospitals'].min() * 111:.2f} km
Avg to Hospitals: {analysis_results['distances_to_hospitals'].mean() * 111:.2f} km
Avg to Districts (all): {analysis_results['distances_to_population'].mean() * 111:.2f} km
Avg to {analysis_results['k_nearest']} nearest districts: {analysis_results['avg_k_nearest_km']:.2f} km


Districts (Strict): {analysis_results['population_in_strict']}
Districts (Coverage): {analysis_results['population_in_coverage']}
"""

    ax5.text(
        0.05, 0.95, results_text,
        fontsize=10, family='monospace',
        verticalalignment='top',
        bbox=dict(
            boxstyle='round',
            facecolor='lightgreen',
            alpha=0.8,
            pad=1.2
        ),
        fontweight='bold'
    )

    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    min_hosp_km = analysis_results['distances_to_hospitals'].min() * 111
    avg_all_pop_km = analysis_results['distances_to_population'].mean() * 111
    avg_k_km = analysis_results['avg_k_nearest_km']
    pop_strict = analysis_results['population_in_strict']

    c1 = min_hosp_km >= CONFIG['MIN_DISTANCE_FROM_HOSPITAL_KM']
    c2 = avg_k_km <= CONFIG['MAX_POPULATION_DISTANCE_KM']
    c3 = pop_strict >= CONFIG['MIN_POPULATION_CENTERS_IN_RANGE']

    color = 'lightgreen' if (c1 and c2 and c3) else 'lightcoral'
    status_text = f"""CONSTRAINTS STATUS


Constraint 1: Min Distance
  {min_hosp_km:.2f} km ≥ {CONFIG['MIN_DISTANCE_FROM_HOSPITAL_KM']} km
  {'PASS' if c1 else 'FAIL'}


Constraint 2: Avg distance to {analysis_results['k_nearest']} nearest districts
  {avg_k_km:.2f} km ≤ {CONFIG['MAX_POPULATION_DISTANCE_KM']} km
  {'PASS' if c2 else 'FAIL'}

  (Avg to ALL districts: {avg_all_pop_km:.2f} km)


Constraint 3: Strict Districts
  {pop_strict} ≥ {CONFIG['MIN_POPULATION_CENTERS_IN_RANGE']}
  {'PASS' if c3 else 'FAIL'}


Overall: {'ALL PASS' if (c1 and c2 and c3) else '✗ FAIL'}
"""

    ax6.text(
        0.05, 0.95, status_text,
        fontsize=10, family='monospace',
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor=color, alpha=0.8, pad=1.2),
        fontweight='bold'
    )

    plt.suptitle('PSO WITH REVISED FITNESS', fontsize=17, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n Visualization saved: {output_path}")
    plt.show()
