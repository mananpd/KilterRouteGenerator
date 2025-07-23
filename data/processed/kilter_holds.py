import pandas as pd
import matplotlib.pyplot as plt
import os
import re


class KilterHolds:
    """
    A class to analyze climbing board data, including holes and placements,
    and visualize specific board layouts with custom hold colors.
    """
    def __init__(self, raw_data_dir):
        """
        Initializes the BoardAnalyzer by loading and preprocessing data.

        Args:
            raw_data_dir (str): The directory containing 'holes.csv' and 'placements.csv'.
        """
        self.raw_data_dir = raw_data_dir
        self.df_holes = None
        self.df_placements = None
        self.merged_df = None
        self._initial_merged_df_state = None
        self._load_and_preprocess_data()

    def _load_and_preprocess_data(self):
        """
        Loads the holes and placements data, performs initial cleaning,
        merges them, and sets a default hold color.
        """
        try:
            holes_csv_path = os.path.join(self.raw_data_dir, 'holes.csv')
            placements_csv_path = os.path.join(self.raw_data_dir, 'placements.csv')

            self.df_holes = pd.read_csv(holes_csv_path)
            self.df_placements = pd.read_csv(placements_csv_path)

            # Drop unnecessary columns and rename for consistency
            self.df_holes = self.df_holes.drop(["mirrored_hole_id", "mirror_group", "name", "product_id"], axis=1)
            self.df_holes = self.df_holes.rename(columns={'id': 'hole_id'})
            self.df_placements = self.df_placements.rename(
                columns={'id': 'placement_id', 'default_placement_role_id': 'default_role_id'}
            )

            # Merge the dataframes on 'hole_id'
            self.merged_df = pd.merge(self.df_holes, self.df_placements, on='hole_id')

            # Rearrange columns and reset index
            rearranged_columns = ["placement_id", "hole_id",
                                  "layout_id", "set_id",
                                  "x", "y",
                                  "default_role_id",
                                  "default_hold_color",
                                  "climb_hold_color",
                                  "default_hold_type",
                                  "climb_hold_type",
                                  "use_frequency"]
            self.merged_df = self.merged_df.reindex(columns=rearranged_columns)
            self.merged_df = self.merged_df.reset_index(drop=True)

            # Initialize default hold colors
            self.merged_df["default_hold_color"] = 'black'
            self.merged_df.loc[self.merged_df["default_role_id"] == 15, "default_hold_color"] = 'gray'
            self._reset_default_colors()

            # Initialize default hold type
            self.merged_df["default_hold_type"] = 'unused'
            self._reset_default_hold_type()

            #store the initial state
            self.merged_df["use_frequency"] = 0
            self._initial_merged_df_state = self.merged_df.copy() # Store a copy of the default state

            print("Data loaded and preprocessed successfully.")
        except FileNotFoundError as e:
            print(f"Error: One or more CSV files not found in {self.raw_data_dir}. {e}")
            self.merged_df = None 
        except Exception as e:
            print(f"An error occurred during data loading or preprocessing: {e}")
            self.merged_df = None

    def _reset_default_colors(self):
        """
        Resets the 'climb_hold_color' column in merged_df to its default state:
        'black' for most, 'gray' for holds with default_role_id == 15.
        This ensures previous custom colors are cleared.
        """
        if self.merged_df is not None:
            self.merged_df["climb_hold_color"] = 'black'
            self.merged_df.loc[self.merged_df["default_role_id"] == 15, "climb_hold_color"] = 'gray'
            print("Climb hold colors reset to default")

    def _reset_default_hold_type(self):
        """
        Resets the 'climb_hold_type' column in merged_df to its default state: 'unusued'
        This ensures previous custom hold type are cleared.
        """
        if self.merged_df is not None:
            self.merged_df["climb_hold_type"] = 'unusued'
            print("Climb hold type reset to default.")        

    @staticmethod
    def parse_p_r_string(text_string, color_map):
        """
        Separates a string like 'p1145r12...' into a pandas DataFrame
        with (placement_id, hold_type) pairs and maps hold types to colors.
        Args:
            text_string (str): The input string containing placement and role information.
                                e.g., 'p1096r12p1159r15...'
            color_map (dict): A dictionary mapping 'r' values (hold_type) to colors.
                              e.g., {'r12': 'green', 'r13': 'blue'}
        Returns:
            pandas.DataFrame: A DataFrame with 'placement_id', 'hold_type', and 'hold_color' columns.
        """
        pattern = r'p(\d{4})(r\d{2})'
        extracted_pairs = re.findall(pattern, text_string)

        df = pd.DataFrame(extracted_pairs, columns=['placement_id', 'hold_type'])
        df['placement_id'] = df['placement_id'].astype(int)
        df['climb_hold_color'] = df['hold_type'].map(color_map)
        return df

    def apply_custom_colors(self, custom_string, custom_color_map):
        """
        Applies custom colors to holds based on a provided string and color map.
        This updates the 'climb_hold_color' column in the main merged DataFrame.
        Args:
            custom_string (str): The string containing custom placement and role info.
            custom_color_map (dict): The color map for custom hold types.
        """
        if self.merged_df is None:
            print("Error: Data not loaded. Cannot apply custom colors.")
            return

        # Step 1: Reset all hold colors to their default state
        self._reset_default_colors()

        # Step 2: Parse the new custom string and apply its colors
        result_df = self.parse_p_r_string(custom_string, custom_color_map)
        color_mapping_series = result_df.set_index('placement_id')['climb_hold_color']

        # Map the new colors.
        self.merged_df['climb_hold_color'] = self.merged_df['placement_id'].map(color_mapping_series).fillna(self.merged_df['climb_hold_color'])
        print("Custom colors applied.")

    def save_default_board_to_csv(self, output_path):
        """
        Saves the board data with default hold colors to a CSV file.
        This method ensures the data is in its default color state before saving.

        Args:
            output_path (str): The full path including filename for the output CSV.
        """
        if self._initial_merged_df_state is None:
            print("Error: Initial data state not available. Cannot save default colors.")
            return

        # Use the stored initial default state for saving
        df_to_save = self._initial_merged_df_state.copy()
        try:
            df_to_save.to_csv(output_path, index=False)
            print(f"Default colors saved to {output_path}")
        except Exception as e:
            print(f"Error saving default colors to CSV: {e}")

    def save_current_board_to_csv(self, output_path):
        """
        Saves the board data with the currently applied hold colors to a CSV file.

        Args:
            output_path (str): The full path including filename for the output CSV.
        """
        if self.merged_df is None:
            print("Error: Data not loaded. Cannot save current colors.")
            return

        df_to_save = self.merged_df[['placement_id', 'hole_id', 'x', 'y', 'hold_color']].copy()
        try:
            df_to_save.to_csv(output_path, index=False)
            print(f"Current colors saved to {output_path}")
        except Exception as e:
            print(f"Error saving current colors to CSV: {e}")

    def get_gecko_board_data(self):
        """
        Filters the merged DataFrame to get data specific to the 'gecko board'.

        Returns:
            pandas.DataFrame: A DataFrame containing data for the gecko board,
                              or None if merged_df is not available.
        """
        if self.merged_df is None:
            print("Error: Data not loaded. Cannot get gecko board data.")
            return None

        # Filter for layout_id 1, placement_id < 4000, and y < 160
        gecko_board = self.merged_df[self.merged_df["layout_id"] == 1]
        gecko_board = gecko_board[gecko_board["placement_id"] < 4000]
        gecko_board = gecko_board[gecko_board["y"] < 160]
        return gecko_board

    def plot_gecko_board(self):
        """
        Generates a scatter plot of the gecko board with hold colors.
        """
        gecko_board = self.get_gecko_board_data()
        if gecko_board is None:
            print("Cannot plot: Gecko board data not available.")
            return

        plt.figure(figsize=(8, 10)) # Adjust figure size for better aspect ratio
        plt.scatter(gecko_board["x"], gecko_board["y"], c=gecko_board["climb_hold_color"], s=50, alpha=0.8)
        plt.xlim(0, 145)
        plt.ylim(0, 160)
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title("Gecko Board Hold Placements")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.gca().set_aspect('equal', adjustable='box') # Ensure equal aspect ratio
        plt.show()

# Example Usage:
if __name__ == "__main__":
    # Create dummy CSV files for demonstration purposes
    dummy_data_dir = 'dummy_data'
    output_data_dir = 'output_data' # Directory for saving output CSVs
    processed_data_dir = 'processed_data' # Directory for processed data
    os.makedirs(dummy_data_dir, exist_ok=True)
    os.makedirs(output_data_dir, exist_ok=True)
    os.makedirs(processed_data_dir, exist_ok=True)


    # Create dummy holes.csv
    holes_data = {
        'id': [1, 2, 3, 4, 5, 6, 7],
        'mirrored_hole_id': [8, 9, 10, 11, 12, 13, 14],
        'mirror_group': [1, 1, 2, 2, 3, 3, 4],
        'name': ['hole_A', 'hole_B', 'hole_C', 'hole_D', 'hole_E', 'hole_F', 'hole_G'],
        'product_id': [101, 102, 103, 104, 105, 106, 107],
        'x': [10, 20, 30, 40, 50, 60, 70],
        'y': [10, 20, 30, 40, 50, 60, 70]
    }
    pd.DataFrame(holes_data).to_csv(os.path.join(dummy_data_dir, 'holes.csv'), index=False)

    # Create dummy placements.csv
    placements_data = {
        'id': [1096, 1159, 1200, 1247, 1249, 1262, 1318, 1369, 1390, 1495, 1498, 1551, 1, 2, 3, 4, 5, 6, 7],
        'hole_id': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 1, 2, 3, 4, 5, 6, 7],
        'layout_id': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1],
        'set_id': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1],
        'default_placement_role_id': [12, 15, 12, 13, 13, 13, 13, 13, 14, 15, 15, 15, 10, 10, 15, 12, 12, 10, 15]
    }
    pd.DataFrame(placements_data).to_csv(os.path.join(dummy_data_dir, 'placements.csv'), index=False)

    # Initialize the KilterHolds analyzer
    analyzer = KilterHolds(raw_data_dir=dummy_data_dir)

    # --- Processing for Training: Example with two hypothetical routes ---
    # Define route data and labels
    routes_data = {
        "route_A": {
            "route_string": 'p1096r12p1200r13p1247r14p1390r15', # Example: r12=start, r13=hand, r14=foot, r15=finish
            "color_map": {
                'r12': 'green',  # Start hold
                'r13': 'blue',   # Hand hold
                'r14': 'orange', # Foot hold
                'r15': 'red'     # Finish hold
            },
            "label": "V5" # Example route grade
        },
        "route_B": {
            "route_string": 'p1159r12p1249r13p1318r13p1498r15',
            "color_map": {
                'r12': 'green',
                'r13': 'blue',
                'r15': 'red'
            },
            "label": "V3"
        }
    }

    all_route_features = {} # To store features for all routes

    for route_name, route_info in routes_data.items():
        print(f"\n--- Processing {route_name} (Label: {route_info['label']}) ---")

        # 1. Apply custom colors for the current route
        analyzer.apply_custom_colors(route_info["route_string"], route_info["color_map"])
        # analyzer.plot_gecko_board() # Uncomment to visualize each route

        # 2. Save the current state (with route colors) to a unique CSV
        route_csv_path = os.path.join(processed_data_dir, f"{route_name}_features.csv")
        analyzer.save_current_colors_to_csv(route_csv_path)

        # 3. Initialize NodeFeatureExtractor with this route's CSV
        extractor = NodeFeatureExtractor(kilter_holds_csv_path=route_csv_path)

        # 4. Retrieve and store features
        route_node_features_dict = extractor.get_node_features_dict()
        route_node_feature_matrix = extractor.get_node_feature_matrix()

        all_route_features[route_name] = {
            "node_features_dict": route_node_features_dict,
            "node_feature_matrix": route_node_feature_matrix,
            "label": route_info["label"]
        }

        print(f"{route_name} Node Features Matrix Shape: {route_node_feature_matrix.shape}")
        print(f"{route_name} Node Features (first entry): {next(iter(route_node_features_dict.values()))}")


    print("\n--- All Route Processing Complete ---")
    print("Example: Features for Route A:")
    print(f"Label: {all_route_features['route_A']['label']}")
    print(f"Matrix shape: {all_route_features['route_A']['node_feature_matrix'].shape}")


    # Clean up dummy data and output data
    os.remove(os.path.join(dummy_data_dir, 'holes.csv'))
    os.remove(os.path.join(dummy_data_dir, 'placements.csv'))
    os.rmdir(dummy_data_dir)
    # Remove all generated route CSVs
    for route_name in routes_data.keys():
        os.remove(os.path.join(processed_data_dir, f"{route_name}_features.csv"))
    os.rmdir(processed_data_dir) # Remove the processed_data directory
    # Note: 'output_data' and its contents from previous examples are not created/removed here
    # as this example focuses on route processing for training.
    # If you run the full previous example and this one in sequence, ensure previous cleanup is handled.
