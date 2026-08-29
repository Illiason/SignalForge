import pytest
import json
from app import app


@pytest.fixture
def client():
    """Create a Flask test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


class TestStatusEndpoint:
    """Tests for the /status endpoint."""

    def test_status_returns_200(self, client):
        """Status endpoint should return 200."""
        response = client.get('/status')
        assert response.status_code == 200

    def test_status_has_required_fields(self, client):
        """Status response should include model_trained and device."""
        response = client.get('/status')
        data = json.loads(response.data)

        assert 'model_trained' in data
        assert 'device' in data
        assert isinstance(data['model_trained'], bool)


class TestCoinsEndpoint:
    """Tests for the /coins endpoint."""

    def test_coins_returns_200(self, client):
        """Coins endpoint should return 200."""
        response = client.get('/coins')
        assert response.status_code == 200

    def test_coins_has_required_fields(self, client):
        """Coins response should include supported_coins and count."""
        response = client.get('/coins')
        data = json.loads(response.data)

        assert 'supported_coins' in data
        assert 'count' in data
        assert isinstance(data['supported_coins'], list)
        assert len(data['supported_coins']) == data['count']

    def test_coins_list_is_correct(self, client):
        """Coins list should have expected cryptocurrencies."""
        response = client.get('/coins')
        data = json.loads(response.data)

        expected = {'Bitcoin', 'Ethereum', 'Solana', 'Cardano', 'Polkadot', 'XRP'}
        assert set(data['supported_coins']) == expected
        assert data['count'] == 6


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""

    def test_predict_missing_json(self, client):
        """POST without JSON body should return 400."""
        response = client.post('/predict')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'error' in data

    def test_predict_missing_news_field(self, client):
        """POST without 'news' field should return 400."""
        response = client.post('/predict',
                              data=json.dumps({'text': 'some news'}),
                              content_type='application/json')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'Missing required field' in data['error']

    def test_predict_empty_news(self, client):
        """POST with empty news string should return 400."""
        response = client.post('/predict',
                              data=json.dumps({'news': ''}),
                              content_type='application/json')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'non-empty string' in data['error']

    def test_predict_whitespace_only_news(self, client):
        """POST with whitespace-only news should return 400."""
        response = client.post('/predict',
                              data=json.dumps({'news': '   \n  '}),
                              content_type='application/json')
        assert response.status_code == 400

    def test_predict_non_string_news(self, client):
        """POST with non-string news field should return 400."""
        response = client.post('/predict',
                              data=json.dumps({'news': 12345}),
                              content_type='application/json')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'non-empty string' in data['error']

    def test_predict_oversized_input(self, client):
        """POST with news > 5000 chars should return 400."""
        huge_news = 'a' * 5001
        response = client.post('/predict',
                              data=json.dumps({'news': huge_news}),
                              content_type='application/json')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'exceeds maximum length' in data['error']

    def test_predict_valid_input_structure(self, client):
        """Valid POST should return 200 and include required fields."""
        response = client.post('/predict',
                              data=json.dumps({'news': 'Bitcoin rally today'}),
                              content_type='application/json')
        # Model may or may not be trained in test env, but input validation passes
        assert response.status_code in [200, 503]
        data = json.loads(response.data)
        assert 'success' in data

    def test_predict_with_coin(self, client):
        """POST with coin parameter should accept supported coins."""
        response = client.post('/predict',
                              data=json.dumps({'news': 'Ethereum upgrade', 'coin': 'Ethereum'}),
                              content_type='application/json')
        assert response.status_code in [200, 503]
        data = json.loads(response.data)
        assert 'success' in data
        if response.status_code == 200:
            assert data['coin'] == 'Ethereum'

    def test_predict_unsupported_coin(self, client):
        """POST with unsupported coin should return 400."""
        response = client.post('/predict',
                              data=json.dumps({'news': 'test', 'coin': 'DogeCoin'}),
                              content_type='application/json')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'Unsupported coin' in data['error']

    def test_predict_default_coin(self, client):
        """POST without coin parameter should default to Bitcoin."""
        response = client.post('/predict',
                              data=json.dumps({'news': 'Market news'}),
                              content_type='application/json')
        if response.status_code == 200:
            data = json.loads(response.data)
            assert data['coin'] == 'Bitcoin'

    def test_predict_at_length_limit(self, client):
        """POST with news exactly at 5000 chars should be accepted."""
        max_news = 'b' * 5000
        response = client.post('/predict',
                              data=json.dumps({'news': max_news}),
                              content_type='application/json')
        # Should pass validation (200 or 503 for model, not 400)
        assert response.status_code in [200, 503]
        data = json.loads(response.data)
        assert 'success' in data

    def test_predict_response_json_structure(self, client):
        """Response should always be valid JSON."""
        response = client.post('/predict',
                              data=json.dumps({'news': 'test'}),
                              content_type='application/json')
        # Should not raise
        data = json.loads(response.data)
        assert isinstance(data, dict)


class TestContentType:
    """Tests for proper HTTP status codes."""

    def test_predict_method_get_not_allowed(self, client):
        """GET /predict should not be allowed."""
        response = client.get('/predict')
        assert response.status_code == 405  # Method Not Allowed

    def test_home_returns_html(self, client):
        """GET / should return HTML content."""
        response = client.get('/')
        assert response.status_code == 200
        # Response should contain HTML (not JSON)
        assert b'<!DOCTYPE' in response.data or b'<html' in response.data or b'<body' in response.data
