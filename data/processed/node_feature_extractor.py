import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import numpy as np # Needed for NodeFeatureExtractor if redefined locally

class NodeFeatureExtractor:
    """
    A class to extract and preprocess node features from Kilter Board hold data.
    It takes the default Kilter Holds CSV as input and generates a feature matrix
    suitable for graph-based machine learning models.
    """

    # Define all possible categories for consistent one-hot encoding
    # These should cover all values across ALL your routes
    ALL_CLIMB_HOLD_TYPES = ['unused', 'foot', 'handFoot', 'start', 'finish']
    ALL_CLIMB_HOLD_COLORS = ['black', 'gray', 'red', 'green', 'blue', 'yellow']

    def __init__(self, kilter_holds_csv_path):
        """
        Initializes the NodeFeatureExtractor by loading the Kilter Holds data
        and performing initial filtering for the Gecko board.

        Args:
            kilter_holds_csv_path (str): The full path to the 'kilterHoldsDefault.csv' file.
        """
        self.kilter_holds_csv_path = kilter_holds_csv_path
        self.kilter_holds_df = None
        self.gecko_holds_df = None
        self.node_features_dict = {}
        self.node_feature_matrix = None
        self.node_feature_df = None
        self._load_and_filter_data()
        self._extract_features()

    def _load_and_filter_data(self):
        """
        Loads the kilter holds data from CSV and filters it to get
        the gecko board specific holds.
        """
        try:
            self.kilter_holds_df = pd.read_csv(self.kilter_holds_csv_path)
            print(f"Loaded data from {self.kilter_holds_csv_path}")

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
        if 'climb_hold_type' in self.gecko_holds_df.columns:
            self.gecko_holds_df['climb_hold_type'] = pd.Categorical(
                self.gecko_holds_df['climb_hold_type'], categories=self.ALL_CLIMB_HOLD_TYPES
            )

        # One-hot encode categorical columns
        categorical_cols_to_encode = ['default_hold_color', 
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