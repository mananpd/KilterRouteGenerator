import os
import sys

current_dir = os.path.dirname(os.getcwd())
sys.path.append(current_dir)

raw_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'raw')
holds = KilterHolds(raw_data_dir)

my_string = 'p1110r15p1150r15p1179r12p1180r12p1183r15p1233r13p1250r13p1319r13p1384r14'
color_map = {
        'r12': 'green',
        'r13': 'blue',
        'r14': 'red',
        'r15': 'yellow'
    }

holds.apply_climb(my_string)

extractor = NodeFeatureExtractor(holds.gecko_board_df)

output_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'nodesssssss')
extractor.node_feature_df.to_csv(output_path, index=False)