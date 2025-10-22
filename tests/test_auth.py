import unittest

from MONet.auth import get_token
from unittest.mock import MagicMock, patch

class TestMONetAuthHelpers(unittest.TestCase):
    
    @patch("requests.post")
    def test_get_token(self, mock_post):
        
        username = "test@maia.se"
        password = "my-secret-password!"
        mock_post.return_value = MagicMock(status_code=200, json=lambda: "test-token")
        token = get_token(username=username, password=password)
        self.assertEqual(token, "test-token")
        mock_post.assert_called_once_with(
            "https://iam.cloud.cbh.kth.se/realms/cloud/protocol/openid-connect/token",
            data={"grant_type": "password", "username": username, "password": password, "client_id": "monailabel-app"}
        )