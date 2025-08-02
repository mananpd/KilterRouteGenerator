import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import numpy as np

# Define NodeFeatureExtractor here for a self-contained example.
class NodeFeatureExtractor:
    """
    A class to extract and preprocess node features from Kilter Board hold data.
    It takes either a DataFrame or a CSV file path as input and generates a feature matrix
    suitable for graph-based machine learning models.
    """

    # Define all possible categories for consistent one-hot encoding
    # These should cover all values across ALL your routes
    ALL_CLIMB_HOLD_TYPES = ['unused', 'foot', 'handFoot', 'start', 'finish']
    ALL_CLIMB_HOLD_COLORS = ['black', 'gray', 'red', 'green', 'blue', 'yellow']
    # If default_role_id also needs fixed categories, define them here:
    # ALL_DEFAULT_ROLE_IDS = [10, 12, 13, 14, 15] # Example values

    def __init__(self, climb_holds_df=None, kilter_holds_csv_path=None):
        """
        Initializes the NodeFeatureExtractor by loading the Kilter Holds data
        and performing initial filtering for the Gecko board.

        Args:
            climb_holds_df (pd.DataFrame, optional): A DataFrame containing climb hold data.
                                                    If provided, data will be loaded from here.
            kilter_holds_csv_path (str, optional): The full path to a CSV file containing Kilter Holds data.
                                                   Used only if climb_holds_df is None.
        """
        self.kilter_holds_csv_path = kilter_holds_csv_path
        self.climb_holds_df_input = climb_holds_df # Store the input DataFrame
        self.kilter_holds_df = None
        self.gecko_holds_df = None
        self.node_features_dict = {}
        self.node_feature_matrix = None
        self.node_feature_df = None # New attribute to store the feature DataFrame
        self._load_and_filter_data()
        self._extract_features()

    def _load_and_filter_data(self):
        """
        Loads the kilter holds data from a DataFrame or CSV and filters it to get
        the gecko board specific holds.
        """
        try:
            if self.climb_holds_df_input is not None:
                self.kilter_holds_df = self.climb_holds_df_input.copy() # Use a copy to avoid modifying original
                print(f"Loaded data from climbing route dataframe")
            elif self.kilter_holds_csv_path is not None:
                self.kilter_holds_df = pd.read_csv(self.kilter_holds_csv_path)
                print(f"Loaded data from {self.kilter_holds_csv_path}")
            else:
                raise ValueError("Either 'climb_holds_df' or 'kilter_holds_csv_path' must be provided.")

            # Filter for gecko board specific holds
            self.gecko_holds_df = self.kilter_holds_df[
                (self.kilter_holds_df["layout_id"] == 1) &
                (self.kilter_holds_df["placement_id"] < 4000) &
                (self.kilter_holds_df["y"] < 160)
            ].copy() # Use .copy() to avoid SettingWithCopyWarning
            print("Filtered data for Gecko board.")

        except FileNotFoundError as e:
            print(f"Error: Kilter Holds CSV file not found at {self.kilter_holds_csv_path}. {e}")
            self.kilter_holds_df = None
            self.gecko_holds_df = None
        except Exception as e:
            print(f"An error occurred during data loading or filtering: {e}")
            self.kilter_holds_df = None
            self.gecko_holds_df = None

    def _extract_features(self):
        """
        Normalizes numerical features, encodes categorical features,
        and constructs the node feature dictionary and matrix.
        Ensures consistent one-hot encoding column count.
        """
        if self.gecko_holds_df is None:
            print("Error: Gecko board data not available. Cannot extract features.")
            return

        # Normalize x and y coordinates
        self.gecko_holds_df['x_normalized'] = (self.gecko_holds_df['x'] - self.gecko_holds_df['x'].min()) / (self.gecko_holds_df['x'].max() - self.gecko_holds_df['x'].min())
        self.gecko_holds_df['y_normalized'] = (self.gecko_holds_df['y'] - self.gecko_holds_df['y'].min()) / (self.gecko_holds_df['y'].max() - self.gecko_holds_df['y'].min())
        print("Normalized 'x' and 'y' coordinates.")

        # Ensure categorical columns have predefined categories for consistent one-hot encoding
        # Apply categories to 'default_hold_color' as well if its values can vary
        if 'default_hold_color' in self.gecko_holds_df.columns:
            # Assuming default_hold_color can only be 'black' or 'gray'
            self.gecko_holds_df['default_hold_color'] = pd.Categorical(
                self.gecko_holds_df['default_hold_color'], categories=['black', 'gray']
            )

        if 'climb_hold_type' in self.gecko_holds_df.columns:
            self.gecko_holds_df['climb_hold_type'] = pd.Categorical(
                self.gecko_holds_df['climb_hold_type'], categories=self.ALL_CLIMB_HOLD_TYPES
            )
        if 'climb_hold_color' in self.gecko_holds_df.columns:
            self.gecko_holds_df['climb_hold_color'] = pd.Categorical(
                self.gecko_holds_df['climb_hold_color'], categories=self.ALL_CLIMB_HOLD_COLORS
            )

        # One-hot encode categorical columns
        categorical_cols_to_encode = [
            'default_hold_color',
            'climb_hold_type']
        
        # Ensure that these columns exist before encoding
        existing_categorical_cols = [col for col in categorical_cols_to_encode if col in self.gecko_holds_df.columns]
        self.gecko_holds_df_encoded = pd.get_dummies(self.gecko_holds_df, columns=existing_categorical_cols, prefix=existing_categorical_cols)
        print(f"One-hot encoded categorical columns: {existing_categorical_cols}")

        # Define feature columns for the final matrix
        # Include 'use_frequency'
        feature_columns = ['x_normalized', 'y_normalized', 'use_frequency'] + \
                          [col for col in self.gecko_holds_df_encoded.columns if any(p in col for p in existing_categorical_cols)]

        # Populate node_features_dict
        for index, row in self.gecko_holds_df_encoded.iterrows():
            hole_id = row['hole_id'] # Use the actual hole_id as the key
            features = row[feature_columns].tolist()
            self.node_features_dict[hole_id] = features
        print("Populated node features dictionary.")

        # Create the final node feature dataframe and matrix
        self.node_feature_df = self.gecko_holds_df_encoded[feature_columns].copy()
        self.node_feature_matrix = self.gecko_holds_df_encoded[feature_columns].values
        print("Created node feature dataframe and matrix.")


    def get_node_features_dict(self):
        """
        Returns the dictionary mapping hole_id to its feature vector.

        Returns:
            dict: A dictionary where keys are hole_ids and values are lists of features.
        """
        return self.node_features_dict

    def get_node_feature_matrix(self):
        """
        Returns the NumPy array representing the node feature matrix.

        Returns:
            numpy.ndarray: The feature matrix where each row is a node's features.
        """
        return self.node_feature_matrix

    def get_gecko_holds_df_encoded(self):
        """
        Returns the DataFrame with normalized coordinates and encoded categorical features.

        Returns:
            pandas.DataFrame: The processed DataFrame.
        """
        return self.gecko_holds_df_encoded
    
    def plot_board(self):
        """
        Generates a scatter plot of the gecko board with hold colors.
        """
        gecko_board = self.node_feature_df.copy()

        if gecko_board is None:
            print("Cannot plot: Gecko board data not available. Run apply_climb first or ensure data loaded.")
            return
        
        # Initialize a default color for all holds
        gecko_board['plot_color'] = 'black'

        # Define conditions and choices for np.select
        conditions = [
            gecko_board['climb_hold_type_foot'] == True,
            gecko_board['climb_hold_type_handFoot'] == True,
            gecko_board['climb_hold_type_start'] == True,
            gecko_board['climb_hold_type_finish'] == True
        ]
        choices = ['yellow', 'blue', 'green', 'red']

        gecko_board['plot_color'] = np.select(conditions, choices, default=gecko_board['plot_color'])
            
        plt.figure(figsize=(8, 10)) # Adjust figure size for better aspect ratio
        # Plot using 'climb_hold_color' for visualization
        plt.scatter(gecko_board["x_normalized"], gecko_board["y_normalized"], c=gecko_board['plot_color'], s=50, alpha=0.8)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title("Gecko Board Hold Placements")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.gca().set_aspect('equal', adjustable='box') # Ensure equal aspect ratio
        plt.show()