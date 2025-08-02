import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from collections import defaultdict
import pandas as pd
from typing import List, Tuple


class ExtractKilterHolds:
    """
    A class to analyze climbing board data, including holes and placements,
    and visualize specific board layouts with custom hold colors. Extract kilter hold data from raw files
    """
    COLOR_MAP = {'r12': 'green', 'r13': 'blue',   # Start, Hand/Foot
                 'r14': 'red', 'r15': 'yellow',  # Finish, Foot
                 'r20': 'green', 'r21': 'blue',   # Start, Hand/Foot
                 'r22': 'red', 'r23': 'yellow',  # Finish, Foot
                 'r24': 'green', 'r25': 'blue',   # Start, Hand/Foot
                 'r26': 'red', 'r27': 'yellow',  # Finish, Foot
                 'r28': 'green', 'r29': 'blue',   # Start, Hand/Foot
                 'r30': 'red', 'r31': 'yellow',  # Finish, Foot
                 }   
    
    ROLE_TYPE_MAP = {'r12': 'start', 'r13': 'handFoot',
                     'r14': 'finish', 'r15': 'foot',
                     'r20': 'start', 'r21': 'handFoot',
                     'r22': 'finish', 'r23': 'foot',
                     'r24': 'start', 'r25': 'handFoot',
                     'r26': 'finish','r27': 'foot',
                     'r28': 'start', 'r29': 'handFoot',
                     'r30': 'finish','r31': 'foot'} 

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
        self.gecko_board_df = None # Current gecko board state
        self._initial_gecko_board_df = None # Initial gecko board state (default colors)
        self._load_and_preprocess_data()


    def _load_and_preprocess_data(self):
        """
        Loads the holes and placements data, performs initial cleaning,
        merges them, and sets default hold properties.
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

            # Initialize default hold colors and types
            self.merged_df["default_hold_color"] = 'black'
            self.merged_df.loc[self.merged_df["default_role_id"] == 15, "default_hold_color"] = 'gray'
            self.merged_df["default_hold_type"] = 'unused' # Default for all holds
            self.merged_df["use_frequency"] = 0 # Initialize use_frequency

            # Initialize climb-specific columns to their default states
            self._reset_to_default_colors() # Initializes 'climb_hold_color'
            self._reset_to_default_hold_type() # Initializes 'climb_hold_type'

            # Store the initial state of the full merged_df
            self._initial_merged_df_state = self.merged_df.copy()

            # Store the initial state of the gecko board (default colors)
            self._initial_gecko_board_df = self.get_gecko_board_data(self._initial_merged_df_state).copy()

            print("Data loaded and preprocessed successfully.")
        except FileNotFoundError as e:
            print(f"Error: One or more CSV files not found in {self.raw_data_dir}. {e}")
            self.merged_df = None
        except Exception as e:
            print(f"An error occurred during data loading or preprocessing: {e}")
            self.merged_df = None

    def _reset_to_default_colors(self):
        """
        Resets the 'climb_hold_color' column in merged_df to its default state:
        'black' for most, 'gray' for holds with default_role_id == 15.
        This ensures previous custom colors are cleared.
        """
        if self.merged_df is not None:
            self.merged_df["climb_hold_color"] = 'black'
            self.merged_df.loc[self.merged_df["default_role_id"] == 15, "climb_hold_color"] = 'gray'
            self.gecko_board_df = None # Invalidate cached gecko board data
            print("Climb hold colors reset to default.")

    def _reset_to_default_hold_type(self):
        """
        Resets the 'climb_hold_type' column in merged_df to its default state: 'unused'
        This ensures previous custom hold types are cleared.
        """
        if self.merged_df is not None:
            self.merged_df["climb_hold_type"] = 'unused'
            self.gecko_board_df = None # Invalidate cached gecko board data
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
        df['climb_hold_color'] = df['hold_type'].map(ExtractKilterHolds.COLOR_MAP)

        # Map 'hold_type' (e.g., 'r12') to 'climb_hold_type' using the class's ROLE_TYPE_MAP
        df['climb_hold_type'] = df['hold_type'].map(ExtractKilterHolds.ROLE_TYPE_MAP)

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
        self._reset_to_default_colors()
        self._reset_to_default_hold_type()

        # Step 2: Parse the new custom string. parse_p_r_string now uses KilterHolds.COLOR_MAP and ROLE_TYPE_MAP
        result_df = self.parse_p_r_string(custom_string)

        # Update climb_hold_color
        color_mapping_series = result_df.set_index('placement_id')['climb_hold_color']
        self.merged_df['climb_hold_color'] = self.merged_df['placement_id'].map(color_mapping_series).fillna(self.merged_df['climb_hold_color'])

        # Update climb_hold_type
        type_mapping_series = result_df.set_index('placement_id')['climb_hold_type'] # Use climb_hold_type from parsed df
        self.merged_df['climb_hold_type'] = self.merged_df['placement_id'].map(type_mapping_series).fillna(self.merged_df['climb_hold_type'])

        # Update the cached gecko_board_df with the newly applied climb data
        self.gecko_board_df = self.get_gecko_board_data(self.merged_df).copy()

        print("Custom colors and types applied.")

    def calculate_hold_frequency_gecko_climbs(self, gecko_climbs_df = None) -> None:      
        holds_frequency = pd.DataFrame(self.gecko_board_df['placement_id'])
        holds_frequency["count_hand"] = 0
        holds_frequency["count_start"] = 0
        holds_frequency["count_finish"] = 0
        holds_frequency["count_foot"] = 0

        holds_frequency = holds_frequency.set_index('placement_id')

        aggregated_counts = defaultdict(int)
        pattern = r'p(\d{4})(r\d{2})'

        for i in range(gecko_climbs_df.shape[0]): 
            # Get the 'frames' string for the current climb
            frames_string = gecko_climbs_df.iloc[i]['frames']
            
            # Extract all hold (ID, type) pairs from this frames string
            holds_used_in_climb = re.findall(pattern, frames_string)
            
            # Inner loop: iterate through holds in the current climb
            for hold_str, hold_type in holds_used_in_climb:
                hold_int = int(hold_str)

                # Increment the corresponding counter in the defaultdict
                if hold_type == 'r12' or hold_type == 'r20' or hold_type == 'r24' or hold_type == 'r28':
                    aggregated_counts[(hold_int, "count_start")] += 1
                elif hold_type == 'r13' or hold_type == 'r21' or hold_type == 'r25' or hold_type == 'r29':
                    aggregated_counts[(hold_int, "count_hand")] += 1
                elif hold_type == 'r14' or hold_type == 'r22' or hold_type == 'r26' or hold_type == 'r30':
                    aggregated_counts[(hold_int, "count_finish")] += 1
                elif hold_type == 'r15' or hold_type == 'r23' or hold_type == 'r27' or hold_type == 'r32':
                    aggregated_counts[(hold_int, "count_foot")] += 1
                else:
                    print(f"Warning: Unhandled hold_type '{hold_type}' for hold '{hold_str}'.")

            if aggregated_counts:
                records = []
                for (placement_id, count_type), count in aggregated_counts.items():
                    records.append({'placement_id': placement_id, 'count_type': count_type, 'count': count})

                df_temp_counts = pd.DataFrame(records)

                # Pivot the temporary DataFrame to match the structure of holds_frequency
                df_new_counts_pivoted = df_temp_counts.pivot_table(
                    index='placement_id',
                    columns='count_type',
                    values='count',
                    fill_value=0 # Fill missing count types for a given placement_id with 0
                )
                
                # Step 3: Update holds_frequency with the new counts
                # The .update() method efficiently updates existing rows/columns based on index alignment.
                holds_frequency.update(df_new_counts_pivoted)
                return holds_frequency

            else:
                print("No holds found in the specified climbs, holds_frequency remains unchanged.")

                def save_default_board_to_csv(self, output_path):
                    """
                    Saves the board data with default hold colors and types to a CSV file.
                    This method ensures the data is in its initial default state before saving.

                    Args:
                        output_path (str): The full path including filename for the output CSV.
                    """
                    if self._initial_gecko_board_df is None:
                        print("Error: Initial gecko board data not available. Cannot save default board.")
                        return

                    df_to_save = self._initial_gecko_board_df[['placement_id', 'hole_id', 'layout_id', 'set_id',
                                                            'x', 'y', 'default_role_id',
                                                            'default_hold_color', 'climb_hold_color',
                                                            'default_hold_type', 'climb_hold_type',
                                                            'use_frequency']].copy()

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
        if self.gecko_board_df is None:
            print("Error: Current gecko board data not available. Cannot save current board.")
            return

        df_to_save = self.gecko_board_df[[
            'placement_id', 'hole_id', 'layout_id', 'set_id',
            'x', 'y', 'default_role_id',
            'default_hold_color', 'climb_hold_color',
            'default_hold_type', 'climb_hold_type',
            'use_frequency'
        ]].copy()

        try:
            df_to_save.to_csv(output_path, index=False)
            print(f"Current board data saved to {output_path}")
        except Exception as e:
            print(f"Error saving current board data to CSV: {e}")

    def get_gecko_board_data(self, holds_df):
        """
        Filters the given DataFrame to get data specific to the 'gecko board'.

        Args:
            holds_df (pd.DataFrame): The DataFrame to filter (e.g., self.merged_df).

        Returns:
            pandas.DataFrame: A DataFrame containing data for the gecko board,
                              or None if holds_df is None.
        """
        if holds_df is None:
            print("Error: Input DataFrame is None. Cannot get gecko board data.")
            return None

        # Filter for layout_id 1, placement_id < 4000, and y < 160
        gecko_board = holds_df[holds_df["layout_id"] == 1]
        gecko_board = gecko_board[gecko_board["placement_id"] < 4000]
        gecko_board = gecko_board[gecko_board["y"] < 160]
        return gecko_board.copy() 

    def plot_gecko_board(self):
        """
        Generates a scatter plot of the gecko board with hold colors.
        """
        gecko_board = self.gecko_board_df # Use the cached gecko_board_df
        if gecko_board is None:
            print("Cannot plot: Gecko board data not available. Run apply_climb first or ensure data loaded.")
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

