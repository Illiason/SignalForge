import pytest
import pandas as pd
from arnn_model import BitcoinPricePredictor


@pytest.fixture
def predictor():
    """Create a predictor instance for testing."""
    return BitcoinPricePredictor(device='cpu')


class TestRuleBasedPredict:
    """Tests for the deterministic rule-based fallback prediction."""

    def test_positive_sentiment(self, predictor):
        """Positive keywords should predict UP."""
        news = "Bitcoin ETF approval and adoption by major institutions rally the market"
        result = predictor.rule_based_predict(news)

        assert result['predicted_direction'] == 'UP'
        assert result['confidence'] >= 60
        assert 'UP' in result['probabilities']

    def test_negative_sentiment(self, predictor):
        """Negative keywords should predict DOWN."""
        news = "Regulatory ban and security hack crash the crypto market"
        result = predictor.rule_based_predict(news)

        assert result['predicted_direction'] == 'DOWN'
        assert result['confidence'] >= 60
        assert 'DOWN' in result['probabilities']

    def test_neutral_sentiment(self, predictor):
        """Mixed or no keywords should predict FLAT."""
        news = "The weather is nice today and Bitcoin exists"
        result = predictor.rule_based_predict(news)

        assert result['predicted_direction'] == 'FLAT'
        assert result['confidence'] == 55

    def test_empty_input(self, predictor):
        """Empty input should default to FLAT with neutral confidence."""
        result = predictor.rule_based_predict("")

        assert result['predicted_direction'] == 'FLAT'
        assert result['confidence'] == 55

    def test_case_insensitive(self, predictor):
        """Keywords should match regardless of case."""
        news = "BITCOIN ETF APPROVAL"
        result = predictor.rule_based_predict(news)

        assert result['predicted_direction'] == 'UP'

    def test_result_structure(self, predictor):
        """Result should always have required keys."""
        news = "Bitcoin news"
        result = predictor.rule_based_predict(news)

        assert 'predicted_direction' in result
        assert 'probabilities' in result
        assert 'confidence' in result
        assert result['confidence'] > 0


class TestCleanData:
    """Tests for data cleaning methods."""

    def test_clean_price_column(self, predictor):
        """Price column should strip commas and convert to float."""
        df = pd.DataFrame({
            'Price': ['1,234.56', '5,678.90', '10000.00']
        })

        cleaned = predictor.clean_data(df.copy())

        assert cleaned['Price'].dtype in ['float64', 'float32']
        assert cleaned['Price'].iloc[0] == 1234.56
        assert cleaned['Price'].iloc[1] == 5678.90

    def test_clean_change_percent_column(self, predictor):
        """Change % column should strip % sign and convert to float."""
        df = pd.DataFrame({
            'Change %': ['-1.38%', '+2.50%', '0.00%']
        })

        cleaned = predictor.clean_data(df.copy())

        assert cleaned['Change %'].dtype in ['float64', 'float32']
        assert cleaned['Change %'].iloc[0] == -1.38
        assert cleaned['Change %'].iloc[1] == 2.50

    def test_clean_data_missing_columns(self, predictor):
        """clean_data should handle dataframes without Price/Change % columns."""
        df = pd.DataFrame({
            'News': ['Bitcoin rally', 'Market crash']
        })

        cleaned = predictor.clean_data(df.copy())

        # Should return unchanged dataframe
        assert 'News' in cleaned.columns
        assert len(cleaned) == 2


class TestCreatePriceDirectionCategories:
    """Tests for price direction categorization."""

    def test_up_direction(self, predictor):
        """Change > 0.5% should be UP."""
        df = pd.DataFrame({
            'Change %': [0.51, 1.0, 5.0, 100.0]
        })

        categorized = predictor.create_price_direction_categories(df.copy())

        assert all(categorized['price_direction'] == 'UP')

    def test_down_direction(self, predictor):
        """Change < -0.5% should be DOWN."""
        df = pd.DataFrame({
            'Change %': [-0.51, -1.0, -5.0, -100.0]
        })

        categorized = predictor.create_price_direction_categories(df.copy())

        assert all(categorized['price_direction'] == 'DOWN')

    def test_flat_direction(self, predictor):
        """Change between -0.5% and +0.5% should be FLAT."""
        df = pd.DataFrame({
            'Change %': [-0.5, -0.1, 0.0, 0.1, 0.5]
        })

        categorized = predictor.create_price_direction_categories(df.copy())

        assert all(categorized['price_direction'] == 'FLAT')

    def test_mixed_directions(self, predictor):
        """Mixed data should categorize correctly."""
        df = pd.DataFrame({
            'Change %': [2.0, -1.0, 0.2, 5.0, -0.1]
        })

        categorized = predictor.create_price_direction_categories(df.copy())

        assert categorized['price_direction'].iloc[0] == 'UP'
        assert categorized['price_direction'].iloc[1] == 'DOWN'
        assert categorized['price_direction'].iloc[2] == 'FLAT'
        assert categorized['price_direction'].iloc[3] == 'UP'
        assert categorized['price_direction'].iloc[4] == 'FLAT'

    def test_missing_change_column(self, predictor):
        """Without Change % column, should default to FLAT."""
        df = pd.DataFrame({
            'Price': [100, 200, 300]
        })

        categorized = predictor.create_price_direction_categories(df.copy())

        assert 'price_direction' in categorized.columns
        assert all(categorized['price_direction'] == 'FLAT')
