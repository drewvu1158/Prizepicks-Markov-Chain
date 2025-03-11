import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
import requests
from datetime import datetime, timedelta
from nba_api.stats.endpoints import playergamelog, commonplayerinfo, LeagueGameFinder
from nba_api.stats.static import players
import time

class BasketballMarkovChain:
    def __init__(self, n_states=5):
        self.n_states = n_states
        self.transition_matrix = None
        self.state_boundaries = None
        self.sequence_history = None
    
    def _get_state(self, value):
        """Determine which state a value belongs to"""
        for i, (lower, upper) in enumerate(self.state_boundaries):
            if lower <= value <= upper:
                return i
        return 0
    
    def fit(self, sequence):
        """
        Fit the Markov chain using full 10-game sequence
        sequence: list of point values from games (most recent first)
        """
        self.sequence_history = sequence
        
        # Create state boundaries based on the sequence
        flat_seq = np.array(sequence)
        percentiles = np.linspace(0, 100, self.n_states + 1)
        boundaries = np.percentile(flat_seq, percentiles)
        self.state_boundaries = [
            (boundaries[i], boundaries[i+1]) 
            for i in range(len(boundaries)-1)
        ]
        
        # Initialize transition matrix
        self.transition_matrix = np.zeros((self.n_states, self.n_states))
        
        # Count transitions between states through the full sequence
        for i in range(len(sequence) - 1):
            current_state = self._get_state(sequence[i])
            next_state = self._get_state(sequence[i + 1])
            self.transition_matrix[current_state][next_state] += 1
        
        # Convert to probabilities
        row_sums = self.transition_matrix.sum(axis=1)
        for i in range(self.n_states):
            if row_sums[i] > 0:
                self.transition_matrix[i] = self.transition_matrix[i] / row_sums[i]
    
    def predict_next(self, n_simulations=1000):
        """
        Predict next value using the current sequence
        """
        current_state = self._get_state(self.sequence_history[0])  # Most recent game
        predictions = []
        
        for _ in range(n_simulations):
            next_state = np.random.choice(self.n_states, p=self.transition_matrix[current_state])
            lower, upper = self.state_boundaries[next_state]
            predicted_value = np.random.uniform(lower, upper)
            predictions.append(predicted_value)
        
        return predictions

    def get_state_analysis(self):
        """Analyze the states and transitions"""
        analysis = []
        for i, (lower, upper) in enumerate(self.state_boundaries):
            games_in_state = sum(1 for p in self.sequence_history if lower <= p <= upper)
            analysis.append(f"State {i}: {lower:.1f}-{upper:.1f} points ({games_in_state} games)")
        return analysis

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
    
    # Collect exactly 10 games
    points_data = []
    for i in range(10):
        while True:
            try:
                line = input(f"Game {i+1}: ")
                stats = line.strip().split()
                points = int(stats[13])
                points_data.append(points)
                break
            except:
                print("Error: Please enter valid stats")
    
    # Initialize and fit Markov chain with full 10-game sequence
    markov_chain = BasketballMarkovChain(n_states=5)
    markov_chain.fit(points_data)
    
    # Get over/under line
    while True:
        try:
            over_under_line = float(input("\nEnter the over/under line: "))
            break
        except ValueError:
            print("Please enter a valid number")
    
    # Get number of simulations
    n_sims = int(input("Enter number of simulations to run (recommended 1000-10000): "))
    
    # Make predictions
    predictions = markov_chain.predict_next(n_sims)
    
    # Calculate probabilities
    over_count = sum(1 for p in predictions if p > over_under_line)
    over_prob = over_count / len(predictions)
    
    print(f"\nMarkov Chain Analysis for {player_name}")
    print("\nLast 10 games (most recent first):")
    for i, points in enumerate(points_data):
        print(f"Game {i+1}: {points} points")
    
    print("\nState Analysis:")
    for state_info in markov_chain.get_state_analysis():
        print(state_info)
    
    print("\nTransition Matrix:")
    for i in range(markov_chain.n_states):
        print(f"State {i} transitions: {markov_chain.transition_matrix[i]}")
    
    print(f"\nPrediction Summary:")
    print(f"Average predicted points: {np.mean(predictions):.1f}")
    print(f"Median predicted points: {np.median(predictions):.1f}")
    print(f"Standard deviation: {np.std(predictions):.1f}")
    
    print(f"\nOver/Under {over_under_line}:")
    print(f"Over probability: {over_prob:.2%}")
    print(f"Under probability: {1-over_prob:.2%}")
    
    # Calculate trend
    recent_trend = np.mean(points_data[:3]) - np.mean(points_data[3:6])
    print(f"\nRecent Trend: {recent_trend:+.1f} points (comparing last 3 vs previous 3 games)")
    
    # Confidence calculation based on prediction consistency and state transitions
    confidence = 1 - (np.std(predictions) / np.mean(predictions))
    print(f"Prediction confidence: {confidence:.1%}")
    
    # Enhanced recommendation system
    print("\nRecommendation:")
    if confidence > 0.7:
        if over_prob > 0.6:
            print(f"Strong OVER ({over_prob:.1%} probability)")
            print(f"Confidence is high ({confidence:.1%})")
            if recent_trend > 0:
                print("Positive scoring trend supports this prediction")
        elif over_prob < 0.4:
            print(f"Strong UNDER ({1-over_prob:.1%} probability)")
            print(f"Confidence is high ({confidence:.1%})")
            if recent_trend < 0:
                print("Negative scoring trend supports this prediction")
        else:
            print("No strong recommendation despite high confidence")
    else:
        print(f"Low confidence prediction ({confidence:.1%}) - proceed with caution")
        print(f"Over probability: {over_prob:.1%}")
        print(f"Recent trend: {recent_trend:+.1f} points")
