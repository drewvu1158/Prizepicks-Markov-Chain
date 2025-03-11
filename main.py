import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
import requests
from datetime import datetime, timedelta
from nba_api.stats.endpoints import playergamelog, commonplayerinfo, LeagueGameFinder
from nba_api.stats.static import players
import time

class MarkovChainPredictor:
    def __init__(self, n_states=5):
        self.n_states = n_states
        self.discretizer = KBinsDiscretizer(n_bins=n_states, encode='ordinal', strategy='quantile')
        self.transition_matrix = None
        self.state_ranges = None
        
    def fit(self, sequence):
        """
        Fit Markov chain transition matrix using the sequence
        """
        # Convert sequence into discrete states
        sequence_array = np.array(sequence).reshape(-1, 1)
        discrete_states = self.discretizer.fit_transform(sequence_array)
        
        # Calculate state ranges for interpretation
        self.state_ranges = []
        bin_edges = self.discretizer.bin_edges_[0]
        for i in range(len(bin_edges) - 1):
            self.state_ranges.append((bin_edges[i], bin_edges[i + 1]))
        
        # Create transition matrix
        self.transition_matrix = np.zeros((self.n_states, self.n_states))
        
        # Count transitions between states
        for i in range(len(discrete_states) - 1):
            current_state = int(discrete_states[i])
            next_state = int(discrete_states[i + 1])
            self.transition_matrix[current_state][next_state] += 1
        
        # Convert counts to probabilities
        row_sums = self.transition_matrix.sum(axis=1)
        self.transition_matrix = np.divide(self.transition_matrix, 
                                         row_sums[:, np.newaxis], 
                                         where=row_sums[:, np.newaxis]!=0)
        
    def predict_next_state(self, current_value):
        """
        Predict next state using Markov chain
        """
        current_state = int(self.discretizer.transform([[current_value]])[0])
        next_state = np.random.choice(self.n_states, p=self.transition_matrix[current_state])
        return np.random.uniform(self.state_ranges[next_state][0],
                               self.state_ranges[next_state][1])

def evaluate_over_under(points_chain, minutes_chain, fg_chain, three_chain,
                       current_points, current_minutes, current_fg, current_three,
                       line, n_simulations=1000):
    """
    Evaluate over/under probability using multiple Markov chains
    """
    over_count = 0
    
    for _ in range(n_simulations):
        # Use Markov chains to predict next game stats
        pred_points = points_chain.predict_next_state(current_points)
        pred_minutes = minutes_chain.predict_next_state(current_minutes)
        pred_fg = fg_chain.predict_next_state(current_fg)
        pred_three = three_chain.predict_next_state(current_three)
        
        # Weight the prediction (points-based and efficiency-based)
        points_weight = 0.6
        efficiency_weight = 0.4
        
        efficiency_points = (pred_minutes/48) * (  # Scale by minutes
            (pred_fg/100 * 40) +    # Expected 2pt points
            (pred_three/100 * 30)    # Expected 3pt points
        )
        
        final_prediction = (points_weight * pred_points + 
                          efficiency_weight * efficiency_points)
        
        if final_prediction > line:
            over_count += 1
    
    return over_count / n_simulations

def parse_stat_line(stat_line):
    """Parse a single line of stats"""
    stats = stat_line.strip().split()
    
    minutes = float(stats[0])
    fg_made, fg_attempted = map(int, stats[1].split('-'))
    fg_pct = float(stats[2])
    three_made, three_attempted = map(int, stats[3].split('-'))
    three_pct = float(stats[4])
    points = int(stats[13])
    
    return {
        'MIN': minutes,
        'FG%': fg_pct,
        '3P%': three_pct,
        'PTS': points
    }

def fetch_player_stats(player_name):
    """
    Fetch current player stats using NBA API
    """
    # Find player
    player_info = players.find_players_by_full_name(player_name)
    if not player_info:
        print(f"No player found with name: {player_name}")
        return None, None
    
    player_id = player_info[0]['id']
    
    # Add delay to avoid API rate limiting
    time.sleep(1)
    
    try:
        # Get current season game logs
        gamelog = playergamelog.PlayerGameLog(
            player_id=player_id,
            season='2023-24',  # Current season
            season_type_all_star='Regular Season'
        )
        df = gamelog.get_data_frames()[0]
        
        # Get player details
        player_details = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
        details_df = player_details.get_data_frames()[0]
        
        # Get next game
        team_id = details_df['TEAM_ID'].iloc[0]
        
        # Find upcoming games
        today = datetime.now()
        future_date = today + timedelta(days=7)  # Look ahead 7 days
        
        game_finder = LeagueGameFinder(
            team_id_nullable=team_id,
            date_from_nullable=today.strftime('%m/%d/%Y'),
            date_to_nullable=future_date.strftime('%m/%d/%Y'),
            season_nullable='2023-24',
            league_id_nullable='00'
        )
        upcoming_games = game_finder.get_data_frames()[0]
        
        return df, details_df, upcoming_games
        
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        return None, None, None

if __name__ == "__main__":
    player_name = input("Enter player name: ")
    print("\nEnter 10 games of stats (most recent first)")
    print("Format: MIN FG FG% 3PT 3P% FT FT% REB AST BLK STL PF TO PTS")
    print("Example: 40 7-19 36.8 1-7 14.3 2-4 50.0 4 2 1 1 2 2 17")
    
    games_data = []
    for i in range(10):
        while True:
            try:
                line = input(f"Game {i+1}: ")
                game_stats = parse_stat_line(line)
                games_data.append(game_stats)
                break
            except Exception as e:
                print(f"Error parsing line. Please ensure format matches example.")
    
    # Extract sequences for the model
    minutes = [game['MIN'] for game in games_data]
    fg_pcts = [game['FG%'] for game in games_data]
    three_pcts = [game['3P%'] for game in games_data]
    points = [game['PTS'] for game in games_data]
    
    # Create separate Markov chains for each metric
    points_chain = MarkovChainPredictor(n_states=5)
    minutes_chain = MarkovChainPredictor(n_states=5)
    fg_chain = MarkovChainPredictor(n_states=5)
    three_chain = MarkovChainPredictor(n_states=5)
    
    # Fit each chain
    points_chain.fit(points)
    minutes_chain.fit(minutes)
    fg_chain.fit(fg_pcts)
    three_chain.fit(three_pcts)
    
    # Get over/under line
    while True:
        try:
            over_under_line = float(input("\nEnter the over/under line: "))
            break
        except ValueError:
            print("Please enter a valid number")
    
    # Calculate probability using Markov chains
    over_prob = evaluate_over_under(
        points_chain, minutes_chain, fg_chain, three_chain,
        points[0], minutes[0], fg_pcts[0], three_pcts[0],
        over_under_line
    )
    
    print(f"\nPrediction for {player_name}:")
    print(f"Last game: {points[0]} points in {minutes[0]} minutes")
    print(f"Last game shooting: FG: {fg_pcts[0]}%, 3P: {three_pcts[0]}%")
    print(f"\nOver/Under line: {over_under_line}")
    print(f"Probability of Over: {over_prob:.2%}")
    print(f"Probability of Under: {(1-over_prob):.2%}")
    
    # Get user's bet choice
    while True:
        choice = input("\nWould you like to bet Over or Under? (O/U): ").upper()
        if choice in ['O', 'U']:
            break
        print("Please enter 'O' for Over or 'U' for Under")
    
    probability = over_prob if choice == 'O' else (1-over_prob)
    bet_type = "OVER" if choice == 'O' else "UNDER"
    
    print(f"\nYou chose {bet_type} {over_under_line}")
    print(f"Model probability for your bet: {probability:.2%}")
    
    if probability > 0.55:
        print("Model suggests this might be a good bet!")
    elif probability < 0.45:
        print("Model suggests this might not be a good bet.")
    else:
        print("Model suggests this is a close call.")
