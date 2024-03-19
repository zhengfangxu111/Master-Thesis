import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import glob
import re

######################################################################################################### Data process
### read data
def read_csv_files(files_path):
    files = os.listdir(files_path)
    csv_files = [file for file in files if file.endswith('.csv')]

    dataframes = []

    for file in csv_files:
        file_path = os.path.join(files_path, file)
        
        # Extract values from the filename using regular expressions
        match = re.search(r'participant_(\d+)_scenario_(\d+)_([a-zA-Z])_at_', file)
        if match:
            participant = int(match.group(1))
            scenario = int(match.group(2))
            start_point = match.group(3)
        else:
            print(f"Filename format not recognized: {file}")
            continue
        
        try:
            # Read the CSV file
            df = pd.read_csv(file_path, delimiter=';', encoding='utf-8')
            
            # Add the extracted values as new columns
            df['participant'] = participant
            df['scenario'] = scenario
            df['start_point'] = start_point
            
            # Append the dataframe to the list
            dataframes.append(df)
        except pd.errors.ParserError as e:
            print(f"Error parsing {file}: {e}")

    # Concatenate all dataframes into a single dataframe
    combined_dataframe = pd.concat(dataframes, ignore_index=True)
    
    return combined_dataframe


### detection of wrong log types
def filter_duplicated_taskAccepted(combined_dataframe):
    # Group by 'participant', 'scenario', and 'start_point', count occurrences of 'taskAccepted'
    grouped_data = combined_dataframe[combined_dataframe['taskState'] == 'taskAccepted'].groupby(['participant', 'scenario', 'start_point']).size().reset_index(name='taskAccepted_count')

    # Filter groups where 'taskAccepted' count is not equal to 4
    filtered_groups = grouped_data[grouped_data['taskAccepted_count'] != 4]

    return filtered_groups

def filter_duplicated_checkedIn(combined_dataframe):
    # Fill missing values in 'taskState' column with an empty string
    combined_dataframe['taskState'] = combined_dataframe['taskState'].fillna('')
    
    # Filter rows where 'taskState' column contains 'checkedIn'
    checked_in_data = combined_dataframe[combined_dataframe['taskState'].str.contains('checkedIn')]

    # Group by 'participant', 'scenario', and 'start_point', count occurrences of 'taskState' containing 'checkedIn'
    grouped_checked_in_data = checked_in_data.groupby(['participant', 'scenario', 'start_point']).size().reset_index(name='checkedIn_count')

    # Filter groups where 'taskState' containing 'checkedIn' count is not equal to 4
    filtered_groups = grouped_checked_in_data[grouped_checked_in_data['checkedIn_count'] != 4]

    return filtered_groups

def filter_navigation_groups(combined_dataframe):
    # Fill missing values in 'taskState' column with an empty string
    combined_dataframe['taskState'] = combined_dataframe['taskState'].fillna('')
    
    # Filter rows where 'taskState' column contains 'Navigation'
    navigation_data = combined_dataframe[combined_dataframe['taskState'].str.contains('Navigation')]

    # Group by 'participant', 'scenario', and 'start_point', count occurrences of 'taskState' containing 'Navigation'
    grouped_navigation_data = navigation_data.groupby(['participant', 'scenario', 'start_point']).size().reset_index(name='navigation_count')

    # Filter groups where 'taskState' containing 'Navigation' count is not equal to 4
    filtered_groups = grouped_navigation_data[grouped_navigation_data['navigation_count'] != 4]

    return filtered_groups

### clean up wrong log types
def replace_fourth_taskAccepted(combined_dataframe, participant_number, scenario_number):
    # Filter rows based on participant number and scenario number
    filtered_rows = combined_dataframe[(combined_dataframe['participant'] == participant_number) & 
                                       (combined_dataframe['scenario'] == scenario_number)]

    # Counter to track the number of 'taskAccepted' occurrences
    task_accepted_count = 0

    # Iterate over the rows and replace the fourth occurrence of 'taskAccepted'
    for index, row in filtered_rows.iterrows():
        if row['taskState'] == 'taskAccepted':
            task_accepted_count += 1
            if task_accepted_count == 4:
                # Replace the fourth occurrence of 'taskAccepted' with an empty string
                combined_dataframe.at[index, 'taskState'] = ''

    return combined_dataframe

def replace_third_taskAccepted(combined_dataframe, participant_number, scenario_number):
    filtered_rows = combined_dataframe[(combined_dataframe['participant'] == participant_number) & 
                                       (combined_dataframe['scenario'] == scenario_number)]
    task_accepted_count = 0

    for index, row in filtered_rows.iterrows():
        if row['taskState'] == 'taskAccepted':
            task_accepted_count += 1
            if task_accepted_count == 3:
                combined_dataframe.at[index, 'taskState'] = ''

    return combined_dataframe

def replace_fifth_taskAccepted(combined_dataframe, participant_number, scenario_number):
    filtered_rows = combined_dataframe[(combined_dataframe['participant'] == participant_number) & 
                                       (combined_dataframe['scenario'] == scenario_number)]
    task_accepted_count = 0

    for index, row in filtered_rows.iterrows():
        if row['taskState'] == 'taskAccepted':
            task_accepted_count += 1
            if task_accepted_count == 5:
                combined_dataframe.at[index, 'taskState'] = ''

    return combined_dataframe

def replace_second_taskAccepted(combined_dataframe, participant_number, scenario_number):
    filtered_rows = combined_dataframe[(combined_dataframe['participant'] == participant_number) & 
                                       (combined_dataframe['scenario'] == scenario_number)]
    task_accepted_count = 0

    for index, row in filtered_rows.iterrows():
        if row['taskState'] == 'taskAccepted':
            task_accepted_count += 1
            if task_accepted_count == 2:
                combined_dataframe.at[index, 'taskState'] = ''

    return combined_dataframe


def replace_fifth_checkedIn(combined_dataframe, participant_number, scenario_number):
    # Filter rows based on participant number and scenario number
    filtered_rows = combined_dataframe[(combined_dataframe['participant'] == participant_number) & 
                                       (combined_dataframe['scenario'] == scenario_number)]

    # Counter to track the number of 'checkedIn' occurrences
    checked_in_count = 0

    # Iterate over the rows and replace the fifth occurrence of 'checkedIn'
    for index, row in filtered_rows.iterrows():
        if 'checkedIn' in str(row['taskState']):
            checked_in_count += 1
            if checked_in_count == 5:
                # Replace the fifth occurrence of 'checkedIn' with an empty string
                combined_dataframe.at[index, 'taskState'] = ''

def replace_third_checkedIn(combined_dataframe, participant_number, scenario_number):
    # Filter rows based on participant number and scenario number
    filtered_rows = combined_dataframe[(combined_dataframe['participant'] == participant_number) & 
                                       (combined_dataframe['scenario'] == scenario_number)]

    # Counter to track the number of 'checkedIn' occurrences
    checked_in_count = 0

    # Iterate over the rows and replace the fifth occurrence of 'checkedIn'
    for index, row in filtered_rows.iterrows():
        if 'checkedIn' in str(row['taskState']):
            checked_in_count += 1
            if checked_in_count == 3:
                # Replace the fifth occurrence of 'checkedIn' with an empty string
                combined_dataframe.at[index, 'taskState'] = ''




def replace_debug_string(combined_dataframe, participant_number, scenario_number):
    # Filter rows based on participant number and scenario number
    filtered_rows = combined_dataframe[(combined_dataframe['participant'] == participant_number) & 
                                       (combined_dataframe['scenario'] == scenario_number)]

    # Iterate over the rows and replace the occurrence
    for index, row in filtered_rows.iterrows():
        # Check if the value in 'taskState' is a string
        if isinstance(row['taskState'], str) and 'DEBUG_IGNORE_ME: GameStart::scenario_selected.gameTasks.Count = 1' in row['taskState']:
            # Replace the occurrence
            combined_dataframe.at[index, 'taskState'] = combined_dataframe.at[index, 'taskState'].replace(
                'DEBUG_IGNORE_ME: GameStart::scenario_selected.gameTasks.Count = 1', 'checkedIn:migros', 1)

    return combined_dataframe


def delete_map_interaction(participant_number, scenario_number, combined_dataframe):
    # Fill missing values in 'mapInteractions' column with an empty string
    combined_dataframe['mapInteractions'] = combined_dataframe['mapInteractions'].fillna('')
    
    # Filter rows based on participant number and scenario number
    filtered_rows = combined_dataframe[(combined_dataframe['participant'] == participant_number) & 
                                        (combined_dataframe['scenario'] == scenario_number)]

    # Find the index of the first occurrence of 'mapLog:MapButtonPoiClick+buttonOnStopNavigation'
    first_occurrence_indices = filtered_rows[filtered_rows['mapInteractions'].str.contains('StopNavigation', na=False)].index

    # Update the 'mapInteraction' column for the first occurrence to an empty string
    for index in first_occurrence_indices:
        combined_dataframe.at[index, 'mapInteractions'] = ''

    return combined_dataframe

def delete_map_interaction_second(participant_number, scenario_number, combined_dataframe):
    # Fill missing values in 'mapInteractions' column with an empty string
    combined_dataframe['mapInteractions'] = combined_dataframe['mapInteractions'].fillna('')
    
    # Filter rows based on participant number and scenario number
    filtered_rows = combined_dataframe[(combined_dataframe['participant'] == participant_number) & 
                                        (combined_dataframe['scenario'] == scenario_number)]

    # Find the indices of all occurrences of 'StopNavigation'
    stop_navigation_indices = filtered_rows[filtered_rows['mapInteractions'].str.contains('StopNavigation', na=False)].index

    # Keep track of the first occurrence
    first_occurrence_index = None

    # Iterate over the indices and update the second occurrence
    for index in stop_navigation_indices:
        if first_occurrence_index is None:
            first_occurrence_index = index
        else:
            # Update the 'mapInteraction' column for the second occurrence to an empty string
            combined_dataframe.at[index, 'mapInteractions'] = ''

    return combined_dataframe


def remove_duplicate_navigation_targets(combined_dataframe, participant_number, scenario_number):
    # Filter rows based on participant number and scenario number
    filtered_rows = combined_dataframe[(combined_dataframe['participant'] == participant_number) & 
                                       (combined_dataframe['scenario'] == scenario_number)]

    # Extract rows containing setNavigationTarget from taskState column
    navigation_rows = filtered_rows[filtered_rows['taskState'].str.contains('setNavigationTarget:')]

    # Find duplicated navigation targets
    duplicated_navigation_rows = navigation_rows[navigation_rows.duplicated(subset='taskState', keep='first')]

    # Replace the taskState of the second occurrence with an empty string
    for index, row in duplicated_navigation_rows.iterrows():
        if index in navigation_rows.index:
            navigation_rows.loc[index, 'taskState'] = ''

    # Update the original DataFrame with modified navigation rows
    combined_dataframe.loc[navigation_rows.index] = navigation_rows

    return combined_dataframe

def add_phase_column(updated_dataframe):
    # Initialize phase column with an empty string
    updated_dataframe['phase'] = ''

    # Flag to indicate if we are within a navigation phase
    within_navigation_phase = False

    # Iterate over rows
    for index, row in updated_dataframe.iterrows():
        if row['state'] == 'start':
            within_navigation_phase = True
            updated_dataframe.at[index, 'phase'] = 'navigation'
        elif row['state'] == 'end':
            within_navigation_phase = False
            updated_dataframe.at[index, 'phase'] = 'navigation'
        elif within_navigation_phase:
            updated_dataframe.at[index, 'phase'] = 'navigation'

    return updated_dataframe

def mark_start_end_state(combined_dataframe):
    # Fill missing values in 'taskState' and 'mapInteractions' columns with an empty string
    combined_dataframe['taskState'] = combined_dataframe['taskState'].fillna('')
    combined_dataframe['mapInteractions'] = combined_dataframe['mapInteractions'].fillna('')

    # Define a function to check if a row should be marked as 'start'
    def is_start(row):
        return 'Navigation' in row['taskState'] and 'WRONG' not in row['taskState']

    # Define a function to check if a row should be marked as 'end'
    def is_end(row):
        return 'checkedIn' in row['taskState'] or 'StopNavigation' in row['mapInteractions']

    # Apply the functions to create the 'state' column
    combined_dataframe['state'] = combined_dataframe.apply(lambda row: 'start' if is_start(row) else ('end' if is_end(row) else ''), axis=1)

    return combined_dataframe

################ about routename
def count_checkedIn_types(dataframe):
    # Extract the types of checkedIn from the taskState column
    checkedIn_types = dataframe['taskState'].str.extract(r'checkedIn:([^,]+)')

    # Create a dictionary to store the count for each type
    checkedIn_counts = {}

    # Iterate through the extracted types
    for checkedIn_type in checkedIn_types[0].dropna():
        if checkedIn_type in checkedIn_counts:
            checkedIn_counts[checkedIn_type] += 1
        else:
            checkedIn_counts[checkedIn_type] = 1

    # Print the counts for each type
    print("Number of occurrences for each 'checkedIn' type:")
    for checkedIn_type, count in checkedIn_counts.items():
        print(f"{checkedIn_type}: {count} times")

def add_routename_column(updated_dataframe):
    # Initialize the routename column with empty strings
    updated_dataframe['routename'] = ''

    # Variable to hold the routename value from the 'start' state
    start_routename = ''

    # Iterate over rows
    for index, row in updated_dataframe.iterrows():
        if 'setNavigationTarget:' in row['taskState']:
            # Extract the route name after 'setNavigationTarget:'
            route_name = row['taskState'].split('setNavigationTarget:')[1].strip()
            
            # Append the value of the start point and route name
            start_point = f"{row['start_point']}_to_{route_name}"
            start_routename = start_point

        if row['state'] == 'start':
            # Set the start_routename when encountering 'start' state
            updated_dataframe.at[index, 'routename'] = start_routename
        elif row['state'] == 'end':
            # Reset the start_routename when encountering 'end' state
            start_routename = ''
            updated_dataframe.at[index, 'routename'] = ''
        elif start_routename:
            # Assign the start_routename to subsequent rows until 'end' state is encountered
            updated_dataframe.at[index, 'routename'] = start_routename

    return updated_dataframe



######################################################################################################### Plot 1
### plot the points
def byParticipant(participant_number):
    # Set the path to the folder containing CSV files
    folder_path = 'data'  # Replace with the actual path to your folder

    #### set coordinates for map image
    # Load the image
    img = mpimg.imread('map_vector_v3_icons.png')  
    # Extract image dimensions
    img_height, img_width, _ = img.shape
    # Coordinates for the center and bottom-left point
    center_x, center_z = -594.0938, -936.8522
    bottom_left_x, bottom_left_z = -1349.39, -1655.76
    # Calculate the distances from the center to the edges
    dist_to_left = center_x - bottom_left_x
    dist_to_top = center_z - bottom_left_z
    # Create a coordinate system based on the image dimensions
    x = np.linspace(center_x - dist_to_left, center_x + dist_to_left, img_width)
    y = np.linspace(center_z - dist_to_top, center_z + dist_to_top, img_height)
    # Create a meshgrid for the coordinate system
    X, Y = np.meshgrid(x, y)
    # Increase the figure size
    plt.figure(figsize=(20, 16))
    # Plot the image with the coordinate system
    plt.imshow(img, extent=[x[0], x[-1], y[0], y[-1]])



    # Iterate over all CSV files for the specified participant
    csv_pattern = f'participant_{participant_number}_scenario_*.csv'
    csv_files = glob.glob(os.path.join(folder_path, csv_pattern))
     # Define colors for each scenario
    scenario_colors = {'scenario_1': '#008837', 'scenario_2': '#a6dba0', 'scenario_3': '#7b3294', 'scenario_4': '#c2a5cf'}
    # Define labels for each scenario
    scenario_labels = {1: 'LT-Non-A', 2: 'LT-Adapt', 3: 'HT-Non-A', 4: 'HT-Adapt'}
    for csv_path in csv_files:
        # Extract scenario number from the CSV filename
        scenario_number = int(csv_path.split('_')[3])

        if f'scenario_{scenario_number}' in scenario_colors:
            df = pd.read_csv(csv_path, delimiter=';')
            plt.scatter(df['posX'], df['posZ'], s=1, marker='o', label=f'{scenario_number}: {scenario_labels[scenario_number]}',
                        c=scenario_colors[f'scenario_{scenario_number}'])

    # Add labels and title
    plt.xlabel('X-axis (posX)')
    plt.ylabel('Z-axis (posZ)')
    plt.title(f'Trajectories of Participant {participant_number}')

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()



def byScenario(scenario_number):
    # Set the path to the folder containing CSV files
    folder_path = 'data'  # Replace with the actual path to your folder

    #### set coordinates for map image the same as above
    img = mpimg.imread('map_vector_v3_icons.png')  
    img_height, img_width, _ = img.shape
    center_x, center_z = -594.0938, -936.8522
    bottom_left_x, bottom_left_z = -1349.39, -1655.76
    dist_to_left = center_x - bottom_left_x
    dist_to_top = center_z - bottom_left_z
    x = np.linspace(center_x - dist_to_left, center_x + dist_to_left, img_width)
    y = np.linspace(center_z - dist_to_top, center_z + dist_to_top, img_height)
    X, Y = np.meshgrid(x, y)
    plt.figure(figsize=(20, 16))
    plt.imshow(img, extent=[x[0], x[-1], y[0], y[-1]])

    # Define colors for each scenario
    scenario_colors = {'scenario_1': '#008837', 'scenario_2': '#a6dba0', 'scenario_3': '#7b3294', 'scenario_4': '#c2a5cf'}
    # Define labels for each scenario
    scenario_labels = {1: 'LT-Non-A', 2: 'LT-Adapt', 3: 'HT-Non-A', 4: 'HT-Adapt'}
    # Iterate over all CSV files for the specified scenario
    csv_pattern = f'*_scenario_{scenario_number}_*.csv'
    csv_files = glob.glob(os.path.join(folder_path, csv_pattern))

    for csv_path in csv_files:
        df = pd.read_csv(csv_path, delimiter=';')
        plt.scatter(df['posX'], df['posZ'], s=1, marker='o', color=scenario_colors[f'scenario_{scenario_number}'])

    # Add labels and title
    plt.xlabel('X-axis (posX)')
    plt.ylabel('Z-axis (posZ)')
    plt.title(f'Trajectories for Scenario {scenario_number} {scenario_labels[scenario_number]}')

    # Show the plot
    plt.show()


def byPartipantUsing(participant_number):
    folder_path = 'data'  # Replace with the actual path to your folder

    # Load the image
    img = mpimg.imread('map_vector_v3_icons.png')  
    img_height, img_width, _ = img.shape
    center_x, center_z = -594.0938, -936.8522
    bottom_left_x, bottom_left_z = -1349.39, -1655.76
    dist_to_left = center_x - bottom_left_x
    dist_to_top = center_z - bottom_left_z
    x = np.linspace(center_x - dist_to_left, center_x + dist_to_left, img_width)
    y = np.linspace(center_z - dist_to_top, center_z + dist_to_top, img_height)
    X, Y = np.meshgrid(x, y)
    
    plt.figure(figsize=(20, 16))
    plt.imshow(img, extent=[x[0], x[-1], y[0], y[-1]])

    csv_pattern = f'participant_{participant_number}_scenario_*.csv'
    csv_files = glob.glob(os.path.join(folder_path, csv_pattern))

    scenario_colors = {'scenario_1': '#008837', 'scenario_2': '#a6dba0', 'scenario_3': '#7b3294', 'scenario_4': '#c2a5cf'}
    scenario_labels = {1: 'LT-Non-A', 2: 'LT-Adapt', 3: 'HT-Non-A', 4: 'HT-Adapt'}
    
    for csv_path in csv_files:
        scenario_number = int(csv_path.split('_')[3])

        if f'scenario_{scenario_number}' in scenario_colors:
            df = pd.read_csv(csv_path, delimiter=';')

            # Step 1: Plot points where 'mapInteractions' is not empty
            df_filtered = df[df['mapInteractions'].notnull() & (df['mapInteractions'] != '')]
            plt.scatter(df_filtered['posX'], df_filtered['posZ'], s=10, marker='o',
                        label=f'{scenario_number}: {scenario_labels[scenario_number]}',
                        c=scenario_colors[f'scenario_{scenario_number}'])

            # Step 2: Plot points whose 'taskState' contains 'setNavigationTarget:' or 'checkedIn:'
            filter_condition = (
                df['taskState'].str.contains('setNavigationTarget:') |
                df['taskState'].str.contains('checkedIn:')
            )
            df_filtered_task = df[filter_condition]
            plt.scatter(df_filtered_task['posX'], df_filtered_task['posZ'], s=50, marker='o',
                        c='red')

    plt.xlabel('X-axis (posX)')
    plt.ylabel('Z-axis (posZ)')
    plt.title(f'Trajectories of Participant {participant_number}')
    plt.legend()
    plt.show()



    ######################################################################################################### Plot 2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def plot_mapinteraction_points(dataframe):
    # Read the base map image
    img = mpimg.imread('map_vector_v3_icons.png')  
    img_height, img_width, _ = img.shape
    center_x, center_z = -594.0938, -936.8522
    bottom_left_x, bottom_left_z = -1349.39, -1655.76
    dist_to_left = center_x - bottom_left_x
    dist_to_top = center_z - bottom_left_z
    x = np.linspace(center_x - dist_to_left, center_x + dist_to_left, img_width)
    y = np.linspace(center_z - dist_to_top, center_z + dist_to_top, img_height)

    # Plot the base map
    plt.figure(figsize=(20, 16))
    plt.imshow(img, extent=[x[0], x[-1], y[0], y[-1]])

    # Filter points based on phase and mapInteractions
    filtered_df = dataframe[(dataframe['phase'] == 'navigation') & (dataframe['mapInteractions'] != '')]

    # Plot the points on the base map image
    plt.scatter(filtered_df['posX'], filtered_df['posZ'], color='red', s=2, label='Map_interaction Points')
    plt.xlabel('posX')
    plt.ylabel('posZ')
    plt.title('All Map_interaction Points during navigation on Base Map')
    plt.legend()
    plt.show()

def plot_routes_traffic_density(dataframe):
    # Read the base map image
    img = mpimg.imread('map_vector_v3_icons.png')  
    img_height, img_width, _ = img.shape
    center_x, center_z = -594.0938, -936.8522
    bottom_left_x, bottom_left_z = -1349.39, -1655.76
    dist_to_left = center_x - bottom_left_x
    dist_to_top = center_z - bottom_left_z
    x = np.linspace(center_x - dist_to_left, center_x + dist_to_left, img_width)
    y = np.linspace(center_z - dist_to_top, center_z + dist_to_top, img_height)

    # Plot the base map
    plt.figure(figsize=(20, 16))
    plt.imshow(img, extent=[x[0], x[-1], y[0], y[-1]])

    # Filter points based on phase and mapInteractions
    filtered_df_high = dataframe[(dataframe['phase'] == 'navigation') & (dataframe['scenario'].isin([4, 3]))]
    filtered_df_low = dataframe[(dataframe['phase'] == 'navigation') & (dataframe['scenario'].isin([1, 2]))]

    # Plot the points on the base map image
    plt.scatter(filtered_df_high['posX'], filtered_df_high['posZ'], color='#7b3294', s=1, label='Routes in high traffic density')
    plt.scatter(filtered_df_low['posX'], filtered_df_low['posZ'], color='#008837', s=1, label='Routes in low traffic density')
    plt.xlabel('posX')
    plt.ylabel('posZ')
    plt.title('Routes of different traffic density on Base Map')
    plt.legend()
    plt.show()

def plot_maptypes_routes(dataframe):
    # Read the base map image
    img = mpimg.imread('map_vector_v3_icons.png')  
    img_height, img_width, _ = img.shape
    center_x, center_z = -594.0938, -936.8522
    bottom_left_x, bottom_left_z = -1349.39, -1655.76
    dist_to_left = center_x - bottom_left_x
    dist_to_top = center_z - bottom_left_z
    x = np.linspace(center_x - dist_to_left, center_x + dist_to_left, img_width)
    y = np.linspace(center_z - dist_to_top, center_z + dist_to_top, img_height)

    # Plot the base map
    plt.figure(figsize=(20, 16))
    plt.imshow(img, extent=[x[0], x[-1], y[0], y[-1]])

    # Filter points based on phase and mapInteractions
    filtered_df_noad = dataframe[(dataframe['phase'] == 'navigation') & (dataframe['scenario'].isin([1, 3]))]
    filtered_df_ad = dataframe[(dataframe['phase'] == 'navigation') & (dataframe['scenario'].isin([2, 4]))]

    # Plot the points on the base map image
    plt.scatter(filtered_df_noad['posX'], filtered_df_noad['posZ'], color='brown', s=2, label='Routes with no_adaptive map')
    plt.scatter(filtered_df_ad['posX'], filtered_df_ad['posZ'], color='blue', s=2, label='Routes with adaptive map')
    plt.xlabel('posX')
    plt.ylabel('posZ')
    plt.title('Routes of different traffic density on Base Map')
    plt.legend()
    plt.show()


def plot_routes_start_point(dataframe, start_point):
    # Read the base map image
    img = mpimg.imread('map_vector_v3_icons.png')  
    img_height, img_width, _ = img.shape
    center_x, center_z = -594.0938, -936.8522
    bottom_left_x, bottom_left_z = -1349.39, -1655.76
    dist_to_left = center_x - bottom_left_x
    dist_to_top = center_z - bottom_left_z
    x = np.linspace(center_x - dist_to_left, center_x + dist_to_left, img_width)
    y = np.linspace(center_z - dist_to_top, center_z + dist_to_top, img_height)

    # Plot the base map
    plt.figure(figsize=(20, 16))
    plt.imshow(img, extent=[x[0], x[-1], y[0], y[-1]])

    # Filter points based on phase, start_point, and scenario
    filtered_df_1 = dataframe[(dataframe['start_point'] == start_point) & 
                              (dataframe['phase'] == 'navigation') & 
                              (dataframe['scenario'] == 1)] 
    filtered_df_2 = dataframe[(dataframe['start_point'] == start_point) & 
                              (dataframe['phase'] == 'navigation') & 
                              (dataframe['scenario'] == 2)] 
    filtered_df_3 = dataframe[(dataframe['start_point'] == start_point) & 
                              (dataframe['phase'] == 'navigation') & 
                              (dataframe['scenario'] == 3)] 
    filtered_df_4 = dataframe[(dataframe['start_point'] == start_point) & 
                              (dataframe['phase'] == 'navigation') & 
                              (dataframe['scenario'] == 4)] 
    combined_df = pd.concat([filtered_df_1, filtered_df_2, filtered_df_3, filtered_df_4], ignore_index=True)

    # Plot the points on the base map image
    plt.scatter(filtered_df_1['posX'], filtered_df_1['posZ'], color='#008837', s=2, label='Route in scenario1')
    plt.scatter(filtered_df_2['posX'], filtered_df_2['posZ'], color='#a6dba0', s=2, label='Route in scenario2')
    plt.scatter(filtered_df_3['posX'], filtered_df_3['posZ'], color='#7b3294', s=2, label='Route in scenario3')
    plt.scatter(filtered_df_4['posX'], filtered_df_4['posZ'], color='#c2a5cf', s=2, label='Route in scenario4')
    plt.xlabel('posX')
    plt.ylabel('posZ')
    plt.title(f'Routes at start point {start_point}')
    plt.legend()

    # Convert 'posX' and 'posZ' columns to numeric type
    combined_df['posX'] = pd.to_numeric(combined_df['posX'], errors='coerce')
    combined_df['posZ'] = pd.to_numeric(combined_df['posZ'], errors='coerce')

    # Get the range of x and y coordinates of the filtered points
    x_min, x_max = combined_df['posX'].min(), combined_df['posX'].max()
    y_min, y_max = combined_df['posZ'].min(), combined_df['posZ'].max()

    # Add some padding to the limits
    padding = 10  # Adjust this value as needed
    x_min -= padding
    x_max += padding
    y_min -= padding
    y_max += padding

    # Set the limits for x and y axes
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.show()
    plt.savefig(f'route_map_startpoint_{start_point}.png', dpi=300, bbox_inches='tight')

    import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd

def plot_route_routename(dataframe, routename, scenario):
    # Read the base map image
    img = mpimg.imread('map_vector_v3_icons.png')  
    img_height, img_width, _ = img.shape
    center_x, center_z = -594.0938, -936.8522
    bottom_left_x, bottom_left_z = -1349.39, -1655.76
    dist_to_left = center_x - bottom_left_x
    dist_to_top = center_z - bottom_left_z
    x = np.linspace(center_x - dist_to_left, center_x + dist_to_left, img_width)
    y = np.linspace(center_z - dist_to_top, center_z + dist_to_top, img_height)

    # Plot the base map
    plt.figure(figsize=(20, 16))
    plt.imshow(img, extent=[x[0], x[-1], y[0], y[-1]])

    # Filter points based on routename and scenario
    filtered_df = dataframe[(dataframe['routename'] == routename) & 
                            (dataframe['scenario'] == scenario)]
    filtered_df_navigation = dataframe[(dataframe['routename'] == routename) & 
                                        (dataframe['mapInteractions'] != "") & 
                                        (dataframe['scenario'] == scenario)]

    # Plot the points on the base map image
    plt.scatter(filtered_df['posX'], filtered_df['posZ'], color='gray', s=8, label='Route Points')
    plt.scatter(filtered_df_navigation['posX'], filtered_df_navigation['posZ'], color='red', s=8, label='Map interaction Points')

    plt.xlabel('posX')
    plt.ylabel('posZ')
    plt.title(f'{routename} scenario {scenario}')
    plt.legend()

    # Convert 'posX' and 'posZ' columns to numeric type
    filtered_df['posX'] = pd.to_numeric(filtered_df['posX'], errors='coerce')
    filtered_df['posZ'] = pd.to_numeric(filtered_df['posZ'], errors='coerce')

    # Get the range of x and y coordinates of the filtered points
    x_min, x_max = filtered_df['posX'].min(), filtered_df['posX'].max()
    y_min, y_max = filtered_df['posZ'].min(), filtered_df['posZ'].max()

    # Add some padding to the limits
    padding = 10  # Adjust this value as needed
    x_min -= padding
    x_max += padding
    y_min -= padding
    y_max += padding

    # Set the limits for x and y axes
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.show()



