import pandas as pd
import numpy as np
import os
import re
from collections import defaultdict
from typing import List, Dict, Any, Tuple

class ClimbDataProcessor:
    """
    A class to encapsulate the entire data processing workflow for climbing data.
    This includes loading, merging, filtering, and calculating similarity scores.
    """

    def __init__(self, raw_data_dir: str):
        """
        Initializes the processor with the directory containing raw data files.

        Args:
            raw_data_dir (str): The path to the directory with the CSV files.
        """
        self.raw_data_dir = raw_data_dir
        
        # Paths for raw data
        self.climb_stats_csv_path = os.path.join(self.raw_data_dir, 'climb_stats.csv')
        self.climb_cache_csv_path = os.path.join(self.raw_data_dir, 'climb_cache_fields.csv')
        self.climbs_csv_path = os.path.join(self.raw_data_dir, 'climbs.csv')

        # Paths for output data
        self.gecko_climbs_csv_path = os.path.join(self.raw_data_dir, 'gecko_climbs.csv')
        self.all_climb_csv_path = os.path.join(self.raw_data_dir, 'all_climbs.csv')

        # DataFrames
        self.df_all_climb_data = None
        self.df_gecko_climb_data = None
        self.df_similarity = None
        self.similarity_df = None
        self.high_similarity_pairs = None

    def load_and_merge_data(self) -> None:
        """
        Loads the three raw CSV files, renames columns, and merges them into a single DataFrame.
        """
        print("Loading and merging raw data...")
        df_climb_stats = pd.read_csv(self.climb_stats_csv_path)
        df_climb_cache = pd.read_csv(self.climb_cache_csv_path)
        df_climbs = pd.read_csv(self.climbs_csv_path)

        df_climb_stats = df_climb_stats.rename(columns={'climb_uuid': 'uuid'})
        df_climb_cache = df_climb_cache.rename(columns={'climb_uuid': 'uuid'})

        df_all_climb_data = pd.merge(df_climb_stats, df_climb_cache, on='uuid')
        df_all_climb_data = pd.merge(df_all_climb_data, df_climbs, on='uuid')

        df_all_climb_data = df_all_climb_data.rename(columns={'angle_x': 'board_angle', 
                                                               'ascensionist_count_x': 'ascensionist_count_angle',
                                                               'ascensionist_count_y': 'ascensionist_count_total',
                                                               'quality_average_x': 'quality_average_angle',
                                                               'quality_average_y': 'quality_average_total',
                                                               'display_difficulty_x': 'display_difficulty'})
        df_all_climb_data = df_all_climb_data.drop(["fa_username", "fa_at", "setter_username", 'created_at', 
                                                    'angle_y'], axis=1, errors='ignore')
        
        self.df_all_climb_data = df_all_climb_data
        print(f"Initial merged DataFrame shape: {self.df_all_climb_data.shape}")

    def filter_gecko_board(self) -> None:
        """
        Filters the merged DataFrame to isolate specific Gecko Board data,
        applies quality and ascensionist count filters, and drops unnecessary columns.
        """
        if self.df_all_climb_data is None:
            raise ValueError("Data not loaded. Please run load_and_merge_data() first.")

        print("Filtering data for Gecko Board...")

        df_gecko_climb_data = self.df_all_climb_data[self.df_all_climb_data['frames_count'] == 1].copy()
        df_gecko_climb_data = df_gecko_climb_data[df_gecko_climb_data['frames_pace'] == 0].copy()
        
        # GECKO BOARD specific filters
        df_gecko_climb_data = df_gecko_climb_data[df_gecko_climb_data['layout_id'] == 1].copy()
        df_gecko_climb_data = df_gecko_climb_data[df_gecko_climb_data["edge_top"] < 160].copy()
        df_gecko_climb_data = df_gecko_climb_data[df_gecko_climb_data["edge_right"] < 142].copy()
        df_gecko_climb_data = df_gecko_climb_data[df_gecko_climb_data["edge_left"] > 0].copy()
        df_gecko_climb_data = df_gecko_climb_data[df_gecko_climb_data["is_listed"] == 1].copy()

        # Drop unnecessary columns
        columns_to_drop = [
            "benchmark_difficulty", 'description', 'is_nomatch', 'frames_count', 
            'frames_pace', 'display_difficulty_y', 'is_draft',
            'is_listed', 'edge_left', 'edge_right', 'edge_top', 'edge_bottom',
            'setter_id', 'layout_id'
        ]
        df_gecko_climb_data = df_gecko_climb_data.drop(columns_to_drop, axis=1, errors='ignore')

        # Additional filters for similarity calculation
        df_gecko_climb_data = df_gecko_climb_data[df_gecko_climb_data['ascensionist_count_total'] > 2000].copy()
        df_gecko_climb_data = df_gecko_climb_data[df_gecko_climb_data['ascensionist_count_angle'] > 300].copy()
        df_gecko_climb_data = df_gecko_climb_data[df_gecko_climb_data['quality_average_total'] > 2.5].copy()
        df_gecko_climb_data = df_gecko_climb_data[df_gecko_climb_data['quality_average_angle'] > 2.5].copy()

        df_gecko_climb_data = df_gecko_climb_data.reset_index(drop=True)
        
        self.df_gecko_climb_data = df_gecko_climb_data
        print(f"Filtered Gecko Board DataFrame shape: {self.df_gecko_climb_data.shape}")

    @staticmethod
    def extract_and_filter_holds(frames_string: str, pattern: str) -> List[str]:
        """
        Extracts hold IDs from a frames string and filters for specific types.
        """
        extracted_pairs = re.findall(pattern, frames_string)
        relevant_holds = []
        for hold, hold_type in extracted_pairs:
            if hold_type in ['r12', 'r13', 'r14', 'r20', 'r21', 'r22', 'r24', 'r25', 'r26', 'r28', 'r29', 'r30']:
                relevant_holds.append(hold)
        return relevant_holds

    def prepare_similarity_dataframe(self) -> None:
        """
        Prepares a DataFrame for similarity calculation by extracting 'hand_holds'.
        """
        if self.df_gecko_climb_data is None:
            raise ValueError("Gecko board data not filtered. Please run filter_gecko_board() first.")

        print("Preparing DataFrame for similarity analysis...")
        
        df_similarity = self.df_gecko_climb_data.drop(
            ['board_angle', 'ascensionist_count_angle', 'display_difficulty',
             'difficulty_average', 'quality_average_angle', 'ascensionist_count_total',
             'quality_average_total', 'hsm'], axis=1
        ).copy()
        
        df_similarity = df_similarity.drop_duplicates(subset=['uuid'], keep='first').copy()
        df_similarity = df_similarity.reset_index(drop=True)
        
        df_similarity['frames'] = df_similarity['frames'].astype(str).fillna('')
        pattern = r'p(\d{4})(r\d{2})'
        
        df_similarity['hand_holds'] = df_similarity['frames'].apply(
            lambda x: self.extract_and_filter_holds(x, pattern)
        )
        
        self.df_similarity = df_similarity
        print(f"Similarity DataFrame prepared with shape: {self.df_similarity.shape}")

    @staticmethod
    def jaccard_similarity(list1: List[Any], list2: List[Any]) -> float:
        """
        Calculates the Jaccard Similarity between two lists.
        Returns 1.0 if both lists are empty.
        """
        set1 = set(list1)
        set2 = set(list2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        if union == 0:
            return 1.0
        return intersection / union

    def calculate_similarity_matrix(self) -> None:
        """
        Calculates the Jaccard similarity matrix for the 'hand_holds' column.
        """
        if self.df_similarity is None:
            raise ValueError("Similarity DataFrame not prepared. Please run prepare_similarity_dataframe() first.")

        df_to_process = self.df_similarity.copy()

        all_hand_holds_lists = df_to_process['hand_holds'].tolist()
        num_rows = len(all_hand_holds_lists)
        print(f"Starting similarity matrix calculation for {num_rows} rows...")
        
        similarity_matrix = np.zeros((num_rows, num_rows), dtype=np.float32)

        for i in range(num_rows):
            if i % 100 == 0:
                print(f"Processing row {i} of {num_rows}...")
            for j in range(i, num_rows): # Iterate from i to use symmetry
                score = self.jaccard_similarity(all_hand_holds_lists[i], all_hand_holds_lists[j])
                similarity_matrix[i, j] = score
                similarity_matrix[j, i] = score # Assign to the symmetric position
        
        self.similarity_df = pd.DataFrame(
            similarity_matrix,
            index=df_to_process['uuid'],
            columns=df_to_process['uuid']
        )
        print("Similarity matrix calculation complete.")

    def find_high_similarity_pairs(self, threshold: float = 0.95) -> List[Tuple[str, str, float]]:
        """
        Finds pairs of UUIDs with a similarity score above a given threshold.

        Args:
            threshold (float): The similarity score threshold for a pair to be considered.
        
        Returns:
            List[Tuple[str, str, float]]: A sorted list of (uuid1, uuid2, score) tuples.
        """
        if self.similarity_df is None:
            raise ValueError("Similarity matrix not calculated. Please run calculate_similarity_matrix() first.")
        
        print(f"Finding high similarity pairs with threshold >= {threshold:.2f}...")
        uuids = self.similarity_df.index.tolist()
        num_uuids = len(uuids)
        high_similarity_pairs = []

        for i in range(num_uuids):
            for j in range(i + 1, num_uuids):
                uuid1 = uuids[i]
                uuid2 = uuids[j]
                score = self.similarity_df.loc[uuid1, uuid2]

                if score >= threshold:
                    high_similarity_pairs.append((uuid1, uuid2, score))

        high_similarity_pairs.sort(key=lambda x: x[2], reverse=True)
        print(f"Found {len(high_similarity_pairs)} pairs above the threshold.")
        self.high_similarity_pairs = high_similarity_pairs

    def deduplicate_by_quality_score(self) -> None:
        """
        Removes 'duplicate' climbs from the `df_gecko_climb_data` based on high similarity pairs.
        For each pair, the climb with the lower 'quality_average_y' score is removed.
        """
        if self.high_similarity_pairs is None:
            raise ValueError("High similarity pairs not found. Please run find_high_similarity_pairs() first.")

        uuids_to_remove = set()
        
        # Create a dictionary for fast quality score lookup
        quality_map = self.df_gecko_climb_data.set_index('uuid')['quality_average_total'].to_dict()

        for climb_one_uuid, climb_two_uuid, _ in self.high_similarity_pairs:
            # Check if both UUIDs are still present in our working set
            if climb_one_uuid in quality_map and climb_two_uuid in quality_map:
                climb_one_quality = quality_map[climb_one_uuid]
                climb_two_quality = quality_map[climb_two_uuid]

                if climb_one_quality > climb_two_quality:
                    uuids_to_remove.add(climb_two_uuid)
                elif climb_two_quality > climb_one_quality:
                    uuids_to_remove.add(climb_one_uuid)
                else:
                    # If qualities are equal, remove one of them (e.g., the second one in the pair)
                    uuids_to_remove.add(climb_two_uuid)
                    
        # Filter the DataFrame once to remove all marked UUIDs
        initial_count = len(self.df_gecko_climb_data)
        self.df_gecko_climb_data = self.df_gecko_climb_data[~self.df_gecko_climb_data['uuid'].isin(uuids_to_remove)]
        self.df_gecko_climb_data = self.df_gecko_climb_data.reset_index(drop=True)
        final_count = len(self.df_gecko_climb_data)
        
        print(f"\n--- Deduplication by Quality Score ---")
        print(f"Initial number of climbs: {initial_count}")
        print(f"Number of climbs removed: {initial_count - final_count}")
        print(f"Final number of climbs: {final_count}")      

    def drop_too_many_holds(self) -> None:
        """
        Identifies and drops climbs with an excessive number of holds that also have a low difficulty average.
        This is a vectorized and more efficient implementation than a row-by-row loop.
        """
        if self.df_gecko_climb_data is None:
            raise ValueError("Gecko board data not filtered. Please run filter_gecko_board() first.")

        initial_shape = self.df_gecko_climb_data.shape
        print(f"Starting 'drop_too_many_holds' on DataFrame of shape: {initial_shape}")

        pattern = r'p(\d{4})(r\d{2})'
        
        # Vectorized way to create the `many_holds_on_climb` DataFrame
        # Apply the logic to the entire 'frames' column at once

        extracted_data = self.df_gecko_climb_data['frames'].apply(
            lambda x: re.findall(pattern, x)
            )
        
        hand_holds_lists = extracted_data.apply(
            lambda pairs: [hold for hold, hold_type in pairs if hold_type in ['r13', 'r21', 'r25', 'r29']]
            )
        foot_holds_lists = extracted_data.apply(
            lambda pairs: [hold for hold, hold_type in pairs if hold_type in ['r15', 'r23', 'r27', r'31']]
            )
        
        # Create boolean masks for the conditions
        mask_too_many_hand_holds = hand_holds_lists.apply(len) > 12
        mask_too_many_foot_holds = foot_holds_lists.apply(len) > 17
        mask_too_many_total_holds = (hand_holds_lists.apply(len) + foot_holds_lists.apply(len)) > 25
        
        # Combine the masks to find all climbs with too many holds
        mask_many_holds = mask_too_many_hand_holds | mask_too_many_foot_holds | mask_too_many_total_holds
        
        # Get the UUIDs for these climbs
        uuid_many_holds_on_climb = self.df_gecko_climb_data.loc[mask_many_holds, 'uuid']

        # Now, create a final mask to find rows that need to be dropped
        # Condition 1: The row's 'uuid' is in our list of UUIDs with many holds.
        is_in_many_holds_list = self.df_gecko_climb_data['uuid'].isin(uuid_many_holds_on_climb)
        
        # Condition 2: The row's 'difficulty_average' is greater than 13.
        has_high_difficulty = self.df_gecko_climb_data['difficulty_average'] > 13
        
        # Combine both conditions with the '&' operator to find rows that meet BOTH criteria
        rows_to_drop_mask = is_in_many_holds_list & has_high_difficulty
        
        # Get the index of all rows that match the mask
        indices_to_drop = self.df_gecko_climb_data[rows_to_drop_mask].index
        
        # Perform a single, vectorized drop operation on the class's DataFrame
        self.df_gecko_climb_data.drop(indices_to_drop, inplace=True)

        final_shape = self.df_gecko_climb_data.shape
        print(f"Dropped {initial_shape[0] - final_shape[0]} rows.")
        print(f"Final DataFrame shape: {final_shape}")

    def save_dataframes(self) -> None:
        """
        Saves the processed DataFrames to CSV files.
        """
        if self.df_gecko_climb_data is not None:
            self.df_gecko_climb_data.to_csv(self.gecko_climbs_csv_path, index=False)
            print(f"Saved filtered Gecko board data to: {self.gecko_climbs_csv_path}")
        
        if self.df_all_climb_data is not None:
            self.df_all_climb_data.to_csv(self.all_climb_csv_path, index=False)
            print(f"Saved all climb data to: {self.all_climb_csv_path}")
