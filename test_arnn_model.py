import pytest
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
