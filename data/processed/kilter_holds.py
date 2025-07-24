import pandas as pd
import matplotlib.pyplot as plt
import os
import re


class KilterHolds:
    """
    A class to analyze climbing board data, including holes and placements,
    and visualize specific board layouts with custom hold colors.
    """
    COLOR_MAP = {'r12': 'green',
                 'r13': 'blue',
                 'r14': 'red',
                 'r15': 'yellow'}
    
    ROLE_TYPE_MAP = {'r12': 'start',
                     'r13': 'handFoot', 
                     'r14': 'finish', 
                     'r15': 'foot'}

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

            # Initialize climb hold colors and types
            self._reset_default_colors() # Initializes 'climb_hold_color'
            self.merged_df["default_hold_type"] = 'unused'
            self.merged_df["use_frequency"] = 0
            self._reset_default_hold_type() # Initializes 'climb_hold_type'

            # Store the initial state after all defaults are set
            self._initial_merged_df_state = self.merged_df.copy()

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
            print("Climb hold colors reset to default.")

    def _reset_default_hold_type(self):
        """
        Resets the 'climb_hold_type' column in merged_df to its default state: 'unused'
        This ensures previous custom hold types are cleared.
        """
        if self.merged_df is not None:
            self.merged_df["climb_hold_type"] = 'unused'
            print("Climb hold type reset to default.")

    @staticmethod
    def parse_p_r_string(text_string):
        """
        Separates a string like 'p1145r12...' into a pandas DataFrame
        with (placement_id, hold_type) pairs and maps hold types to colors.

        Args:
            text_string (str): The input string containing placement and role information.
                                e.g., 'p1096r12p1159r15...'

        Returns:
            pandas.DataFrame: A DataFrame with 'placement_id', 'hold_type', 'climb_hold_color', and 'climb_hold_type' columns.
        """
        pattern = r'p(\d{4})(r\d{2})'
        extracted_pairs = re.findall(pattern, text_string)

        df = pd.DataFrame(extracted_pairs, columns=['placement_id', 'hold_type'])
        df['placement_id'] = df['placement_id'].astype(int)

        # Map 'hold_type' (e.g., 'r12') to 'climb_hold_color' using the class's COLOR_MAP
        df['climb_hold_color'] = df['hold_type'].map(KilterHolds.COLOR_MAP)

        # Map 'hold_type' (e.g., 'r12') to 'climb_hold_type' using the class's ROLE_TYPE_MAP
        df['climb_hold_type'] = df['hold_type'].map(KilterHolds.ROLE_TYPE_MAP)

        return df[['placement_id', 'hold_type', 'climb_hold_color', 'climb_hold_type']]

    def apply_climb(self, custom_string):
        """
        Applies custom colors and types to holds based on a provided string.
        This updates the 'climb_hold_color' and 'climb_hold_type' columns in the main merged DataFrame.
        Before applying new values, it resets all climb hold colors and types to their defaults.

        Args:
            custom_string (str): The string containing custom placement and role info.
        """
        if self.merged_df is None:
            print("Error: Data not loaded. Cannot apply custom colors.")
            return

        # Step 1: Reset all climb hold colors and types to their default state
        self._reset_default_colors()
        self._reset_default_hold_type()

        # Step 2: Parse the new custom string. parse_p_r_string now uses self.COLOR_MAP and ROLE_TYPE_MAP
        result_df = self.parse_p_r_string(custom_string)

        # Update climb_hold_color
        color_mapping_series = result_df.set_index('placement_id')['climb_hold_color']
        self.merged_df['climb_hold_color'] = self.merged_df['placement_id'].map(color_mapping_series).fillna(self.merged_df['climb_hold_color'])

        # Update climb_hold_type
        type_mapping_series = result_df.set_index('placement_id')['climb_hold_type'] # Use climb_hold_type from parsed df
        self.merged_df['climb_hold_type'] = self.merged_df['placement_id'].map(type_mapping_series).fillna(self.merged_df['climb_hold_type'])

        print("Custom colors and types applied.")

    def save_default_board_to_csv(self, output_path):
        """
        Saves the board data with default hold colors and types to a CSV file.
        This method ensures the data is in its initial default state before saving.

        Args:
            output_path (str): The full path including filename for the output CSV.
        """
        if self._initial_merged_df_state is None:
            print("Error: Initial data state not available. Cannot save default board.")
            return

        # Use the stored initial default state for saving
        df_to_save = self._initial_merged_df_state.copy()
        try:
            df_to_save.to_csv(output_path, index=False)
            print(f"Default board data saved to {output_path}")
        except Exception as e:
            print(f"Error saving default board data to CSV: {e}")

    def save_current_board_to_csv(self, output_path):
        """
        Saves the board data with the currently applied hold colors and types to a CSV file.

        Args:
            output_path (str): The full path including filename for the output CSV.
        """
        if self.merged_df is None:
            print("Error: Data not loaded. Cannot save current board.")
            return

        # Save all relevant columns, including new ones
        df_to_save = self.merged_df[[
            'placement_id', 'hole_id', 'layout_id', 'set_id',
            'x', 'y', 'default_role_id',
            'default_hold_color', 'climb_hold_color',
            'default_hold_type', 'climb_hold_type', 'use_frequency'
        ]].copy()
        try:
            df_to_save.to_csv(output_path, index=False)
            print(f"Current board data saved to {output_path}")
        except Exception as e:
            print(f"Error saving current board data to CSV: {e}")

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
        # Plot using 'climb_hold_color' for visualization
        plt.scatter(gecko_board["x"], gecko_board["y"], c=gecko_board["climb_hold_color"], s=50, alpha=0.8)
        plt.xlim(0, 145)
        plt.ylim(0, 160)
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title("Gecko Board Hold Placements")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.gca().set_aspect('equal', adjustable='box') # Ensure equal aspect ratio
        plt.show()
