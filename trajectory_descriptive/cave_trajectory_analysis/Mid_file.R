library(dplyr)
df <- read.csv("dataframe_allroutename.csv")

# Ensure the DataFrame is sorted by DateTime if it's not already sorted
df <- df %>% arrange(DateTime)

# Add a new column 'startplan' initialized as FALSE
df$startplan <- FALSE

# Find the indices where 'phase' is 'planning'
planning_indices <- which(df$phase == "planning")

# Iterate over these indices
for (i in planning_indices) {
  # Check if the previous row's phase is also 'planning'
  if (i > 1 && df$phase[i - 1] == "planning") {
    # Skip this row since it's not the first row of consecutive planning phases
    next
  } else {
    # This is the first row of consecutive planning phases, mark it as 'startplan'
    df$startplan[i] <- TRUE
  }
}

# Assign 'startplan' to 'state' column where 'startplan' is TRUE
df$state[df$startplan == TRUE] <- 'startplan'

```

```{r setup, include=FALSE}
# Convert DateTime column to POSIXct format
df$DateTime <- as.POSIXct(df$DateTime)

# Initialize an empty list to store the results
result_list <- list()

# Iterate through each participant from 8 to 63 (excluding 11)
for (participant in c(8:10, 12:13, 15:63)) {
  # Iterate through each scenario
  for (scenario in 1:4) {
    # Filter the dataframe for the current participant and scenario
    filtered_df <- df %>% filter(participant == !!participant & scenario == !!scenario)
    
    #___________________________________planning phase 
    # Find the index of 'startplan' and 'start' states
    startplan_index <- which(filtered_df$state == 'startplan')
    start_index <- which(filtered_df$state == 'start')
    
    # ___________time
    if (length(startplan_index) > 0 && length(start_index) > 0) {
      planning_time <- sum(difftime(filtered_df$DateTime[start_index], filtered_df$DateTime[startplan_index], units = "secs"))
    } 
    
    # ___________distance 
    # If both 'startplan' and 'start' states exist
    if (length(startplan_index) > 0 && length(start_index) > 0) {
      # Initialize vectors to store distances and stop counts
      planning_distances <- c()
      planning_stops <- c()
      
      # Iterate over each pair of 'startplan' and 'start' states
      for (i in 1:min(length(startplan_index), length(start_index))) {
        # Extract the subset of planning phase data
        planning_subset <- filtered_df[startplan_index[i]:start_index[i], c("posX", "posZ")]
        
        # Calculate distances between consecutive points
        distances <- sqrt(diff(planning_subset$posX)^2 + diff(planning_subset$posZ)^2)
        
        # Calculate number of stops
        pstops <- sum(duplicated(planning_subset[c("posX", "posZ")]) & !duplicated(planning_subset[c("posX", "posZ")], fromLast = TRUE))
        
        # Append distances and stop counts to respective vectors
        planning_distances <- c(planning_distances, distances)
        planning_stops <- c(planning_stops, pstops)
      }
      
      # Sum the distances to get the total length of planning phase
      planning_length <- sum(planning_distances)
      
      # Sum the stop counts to get the total number of stops in planning phase
      number_planning_stops <- sum(planning_stops)
    }
    
    
    # ___________speed
    planning_time_numeric <- as.numeric(planning_time)
    
    # Calculate planning speed
    planning_speed <- ifelse(!is.na(planning_time_numeric) && planning_time_numeric != 0,
                             planning_length / planning_time_numeric,
                             NA)
    
    
    
    
    #___________________________________navigating phase  
    # Find the index of 'start' and 'end' states
    start_index <- which(filtered_df$state == 'start')
    end_index <- which(filtered_df$state == 'end')
    
    # ___________time
    if (length(start_index) > 0 && length(end_index) > 0) {
      navigating_time <- sum(difftime(filtered_df$DateTime[end_index], filtered_df$DateTime[start_index], units = "secs"))
    }
    
    scenario_time <- ifelse(is.na(planning_time) | is.na(navigating_time), NA, planning_time + navigating_time)
    
    # ___________distance 
    # If both 'start' and 'end' states exist
    if (length(start_index) > 0 && length(end_index) > 0) {
      # Initialize vectors to store distances and stop counts
      navigating_distances <- c()
      navigating_stops <- c()
      
      # Iterate over each pair of 'start' and 'end' states
      for (i in 1:min(length(start_index), length(end_index))) {
        # Extract the subset of navigating phase data
        navigating_subset <- filtered_df[start_index[i]:end_index[i], c("posX", "posZ")]
        
        # Calculate distances between consecutive points
        distances <- sqrt(diff(navigating_subset$posX)^2 + diff(navigating_subset$posZ)^2)
        
        # Calculate number of stops
        nstops <- sum(duplicated(navigating_subset[c("posX", "posZ")]) & !duplicated(navigating_subset[c("posX", "posZ")], fromLast = TRUE))
        
        # Append distances and stop counts to respective vectors
        navigating_distances <- c(navigating_distances, distances)
        navigating_stops <- c(navigating_stops, nstops)
      }
      
      # Sum the distances to get the total length of navigating phase
      navigating_length <- sum(navigating_distances)
      
      # Sum the stop counts to get the total number of stops in navigating phase
      number_navigating_stops <- sum(navigating_stops)
    } 
    #___________speed 
    navigating_time_numeric <- as.numeric(navigating_time)
    
    navigation_speed <- ifelse(!is.na(navigating_time_numeric) && navigating_time_numeric != 0,
                               navigating_length / navigating_time_numeric,
                               NA)
    
    
    
    result <- list(
      participant = participant,
      scenario = scenario,
      planning_time = planning_time,
      planning_length = planning_length,
      navigating_time = navigating_time,
      navigating_length = navigating_length,
      scenario_time = scenario_time,
      planning_speed = planning_speed,
      navigation_speed = navigation_speed,
      number_planning_stops = number_planning_stops,
      number_navigating_stops =number_navigating_stops
      
    )
    # Append the result to the result list
    result_list <- c(result_list, list(result))
  }
}

# Create a dataframe from the result list
result_df <- do.call(rbind, result_list)





filtered_df <- df %>% filter(participant == 15 & scenario == 2)
startplan_index <- which(filtered_df$state == 'startplan')
start_index <- which(filtered_df$state == 'start')

# Find the index of 'start' and 'end' states
start_index <- which(filtered_df$state == 'start')
end_index <- which(filtered_df$state == 'end')

# ___________time
if (length(start_index) > 0 && length(end_index) > 0) {
  navigating_time <- sum(difftime(filtered_df$DateTime[end_index], filtered_df$DateTime[start_index], units = "secs"))
}

