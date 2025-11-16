from config import get_params, print_config_summary
from data_loader import load_data
from core import PSO
from analysis import analyze_results, create_visualization


def main():
    print_config_summary()

    result = load_data()
    if not result:
        print(" Failed to load data. Exiting.")
        return

    districts_df, hospitals_df, districts_coords, hospitals_coords = result

    params = get_params()
    pso = PSO(params, hospitals_coords, districts_coords)

    optimal_location, optimal_fitness = pso.optimize()

    analysis_results = analyze_results(
        optimal_location, params,
        districts_df, hospitals_df,
        districts_coords, hospitals_coords,
        pso
    )

    create_visualization(
        optimal_location, params,
        districts_df, hospitals_df,
        districts_coords, hospitals_coords,
        pso, analysis_results,
        output_path="pso_revised_fitness.png"
    )

    print("\n" + "=" * 110)
    print("✓ OPTIMIZATION COMPLETED (REVISED FITNESS)")
    print("=" * 110)


if __name__ == "__main__":
    main()
