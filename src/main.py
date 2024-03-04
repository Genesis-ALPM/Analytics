

import numpy as np
from numpy.fft import fft
import pandas as pd

class Volatility:

    def __init__(self):
        """
        Initializes the Volatility class with lookup tables and default weights.
        """
        # Define the lookup table for trend analysis
        self.ctable = {
                 ('Uptrend', 'Uptrend', 'Uptrend'): 'Extremely Bullish',
            ('Uptrend', 'Uptrend', 'Sideways'): 'Strong Bullish',
            ('Uptrend', 'Uptrend', 'Downtrend'): 'Strong Bullish',
            
            ('Uptrend', 'Sideways', 'Uptrend'): 'Strong Bullish',
            ('Uptrend', 'Sideways', 'Sideways'): 'Bullish',
            ('Uptrend', 'Sideways', 'Downtrend'): 'Boxed',

            ('Uptrend', 'Downtrend', 'Uptrend'): 'Strong Bullish',
            ('Uptrend', 'Downtrend', 'Sideways'): 'Boxed',
            ('Uptrend', 'Downtrend', 'Downtrend'): 'Bearish',
            
            ('Sideways', 'Uptrend', 'Uptrend'): 'Strong Bullish',
            ('Sideways', 'Uptrend', 'Sideways'): 'Bullish',
            ('Sideways', 'Uptrend', 'Downtrend'): 'Boxed', 

            ('Sideways', 'Sideways', 'Uptrend'): 'Bullish',
            ('Sideways', 'Sideways', 'Sideways'): 'Boxed',
            ('Sideways', 'Sideways', 'Downtrend'): 'Bearish', 

            ('Sideways', 'Downtrend', 'Uptrend'): 'Boxed',
            ('Sideways', 'Downtrend', 'Sideways'): 'Bearish',
            ('Sideways', 'Downtrend', 'Downtrend'): 'Strong Bearish', 

            ('Downtrend', 'Uptrend', 'Uptrend'): 'Strong Bullish',
            ('Downtrend', 'Uptrend', 'Sideways'): 'Boxed',
            ('Downtrend', 'Uptrend', 'Downtrend'): 'Bearish', 

            ('Downtrend', 'Sideways', 'Uptrend'): 'Boxed',
            ('Downtrend', 'Sideways', 'Sideways'): 'Bearish',
            ('Downtrend', 'Sideways', 'Downtrend'): 'Strong Bearish', 

            ('Downtrend', 'Downtrend', 'Uptrend'): 'Bearish',
            ('Downtrend', 'Downtrend', 'Sideways'): 'Strong Bearish',
            ('Downtrend', 'Downtrend', 'Downtrend'): 'Extremely Bearish', 
        }

        # Define the lookup table for market sentiment
        self.sentiment_table = {
            'Extremely Bullish': 50,
            'Strong Bullish': 40,
            'Bullish': 30,
            'Boxed': 50,
            'Bearish': 40,
            'Strong Bearish': 30,
            'Extremely Bearish': 25,
        }

        # Default weights for the three volatility components
        self.ctable_weight = 90
        self.stat_weight = 5
        self.spec_weight = 5

    def get_market_sentiment(self, long_term, medium_term, short_term):
        """
        Calculates market sentiment based on the provided trend components.

        Parameters:
        - long_term: Trend component for the long term.
        - medium_term: Trend component for the medium term.
        - short_term: Trend component for the short term.

        Returns:
        - Market sentiment value or 'Invalid Trend'/'Invalid Sentiment' if input is not valid.
        """
        # Check if the combination exists in the lookup table
        if (long_term, medium_term, short_term) in self.ctable:
            sentiment = self.ctable[(long_term, medium_term, short_term)]
            return self.sentiment_table.get(sentiment, 'Invalid Sentiment')
        else:
            return 'Invalid Trend'

    def compute_weighted_volatility(self, ctable_vol, stat_vol, spec_vol):
        """
        Computes weighted volatility based on the provided components and default weights.

        Parameters:
        - ctable_vol: Volatility value from the ctable component.
        - stat_vol: Volatility value from the statistical component.
        - spec_vol: Volatility value from the spectrum component.

        Returns:
        - List of weighted volatility values.
        """
        # Given initial values and weights
        initial_values = [ctable_vol, stat_vol, spec_vol]
        weights = [self.ctable_weight, self.stat_weight, self.spec_weight]

        # Apply weights to calculate final values
        weighted_volatility = [initial * weight / 100 for initial, weight in zip(initial_values, weights)]
        return weighted_volatility

    def compute_stat_volatility(self, price_series):
        """
        Helper function to calculate volatility based on the mean and standard deviation.

        Parameters:
        - price_series: Series containing the selected price data.

        Returns:
        - Statistical Volatility value.
        """
        price_mean = np.mean(price_series)
        price_std = np.std(price_series)

        volatility = 1 - (price_mean - price_std) / price_mean
        volatility = volatility * 2
        return volatility

    def compute_pds(self, price_series):
        """
        Helper function to calculate spectral volatility based on the power spectral density.

        Parameters:
        - price_series: Series containing the selected price data.

        Returns:
        - Spectral Volatility value.
        """
        myfft = fft(price_series)
        
        n = len(price_series)
        T_long = 1.0
        T_mid = 1.0
        T_short = 1.0

        f_long = np.linspace(0.0, 1.0 / (2.0 * T_long), n // 2)
        pds = 2.0 / n * np.abs(myfft[:n // 2])

        low_freq_idx = f_long < 0.1
        high_freq_idx = f_long >= 0.1

        low_freq_power = np.mean(pds[low_freq_idx])
        high_freq_power = np.mean(pds[high_freq_idx])

        if high_freq_power == 0:
            return 0
        else:
            return low_freq_power / high_freq_power
    
    def compute_spec_volatility(self, price_series):
        """
        Helper function to calculate spectral volatility based on the power spectral density.

        Parameters:
        - price_series: Series containing the selected price data.

        Returns:
        - Spectral Volatility value.
        """
        # Calculate PDS for series[0:end-1]
        pds_prev = self.compute_pds(price_series.iloc[:-1])

        # Calculate PDS for series[1:end]
        pds_current = self.compute_pds(price_series.iloc[1:])

        # Calculate spectral volatility using the formula: (current PDS Ratio)/(previous PDS Ratio) - 1
        spec_vol = pds_current / pds_prev - 1

        return spec_vol



# Testing the updated implementation with new test cases
def test_trend_analyzer():
    analyzer = Volatility()

    # Test case 1: Analyzing trend
    assert analyzer.get_market_sentiment('Uptrend', 'Uptrend', 'Uptrend') == 50

    # Test case 2: Analyzing market sentiment
    assert analyzer.get_market_sentiment('Downtrend', 'Downtrend', 'Sideways') == 30

    # Test case 3: Invalid market sentiment
    assert analyzer.get_market_sentiment('Neutral', 'Neutral', 'Neutral') == 'Invalid Trend'

    print("All test cases passed!")

def test_compute_spec_volatility():
    analyzer = Volatility()

    # Sample price series data
    price_series = pd.Series([10, 12, 15, 11, 14, 16, 18, 20])

    # Call the compute_spec_volatility method
    spec_volatility = analyzer.compute_spec_volatility(price_series)

    # Expected result: (pds_current[1:end] / pds_prev[0:end-1]) - 1
    expected_spec_volatility = (
        analyzer.compute_pds(price_series.iloc[1:]) /
        analyzer.compute_pds(price_series.iloc[:-1]) - 1
    )

    # Check if the computed spec_volatility matches the expected result
    np.testing.assert_array_almost_equal(spec_volatility, expected_spec_volatility)

# Run the test cases
test_trend_analyzer()

test_compute_spec_volatility()