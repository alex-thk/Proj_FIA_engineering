import pandas as pd
import json


def get_top_10_performances(file_path):
    # Read the JSON file into a DataFrame
    df = pd.read_json(file_path, lines=True)

    # Sort the DataFrame by the 'Performance' column in descending order
    df_sorted = df.sort_values(by='Performance', ascending=False)

    # Get the top 10 rows
    top_10_df = df_sorted.head(10)

    return top_10_df


def main():
    # List of input JSON files
    input_files = ['3cluster.json', '4cluster.json', '5cluster.json']
    combined_top_10 = {}

    for file_path in input_files:
        # Get the top 10 performances
        top_10_df = get_top_10_performances(file_path)

        # Add the top 10 rows to the combined dictionary
        combined_top_10[file_path] = top_10_df.to_dict(orient='records')

    # Write the combined top 10 rows to a new JSON file
    output_file_path = 'top_ten_performances.json'
    with open(output_file_path, 'w') as output_file:
        json.dump(combined_top_10, output_file, indent=4)
    print(f'Top 10 performances for each file written to {output_file_path}')


if __name__ == "__main__":
    main()