import unittest
import tempfile
import os
from MONet.auth import get_token, verify_valid_token_exists
from unittest.mock import MagicMock, patch
import json
import time
import shutil

class TestMONetAuthHelpers(unittest.TestCase):
    
    
    def setUp(self):
        self.username = "test@maia.se"
        self.password = "my-secret-password!"
        self.token = "test-token"
        
        self.temp_home_dir = tempfile.mkdtemp()
        os.environ["HOME"] = self.temp_home_dir
        os.makedirs(os.path.join(self.temp_home_dir, ".monet"), exist_ok=True)
        
        with open(os.path.join(self.temp_home_dir, ".monet", f"{self.username}_auth.json"), "w") as f:
            json.dump({"access_token": self.token}, f)
            
    def tearDown(self):
        shutil.rmtree(self.temp_home_dir)
        os.environ.pop("HOME")

    @patch("MONet.auth.requests.post")
    def test_get_token(self, mock_post):
        
        mock_post.return_value = MagicMock(status_code=200, json=lambda: self.token)
        token = get_token(username=self.username, password=self.password)
        self.assertEqual(token, self.token)
        mock_post.assert_called_once_with(
            "https://iam.cloud.cbh.kth.se/realms/cloud/protocol/openid-connect/token",
            data={"grant_type": "password", "username": self.username, "password": self.password, "client_id": "monailabel-app"}
        )

    @patch("MONet.auth.decode")
    def test_verify_valid_token_exists(self, mock_decode):
        mock_decode.return_value = {"exp": time.time() + 3600}
        self.assertTrue(verify_valid_token_exists(username=self.username))
        self.assertFalse(verify_valid_token_exists(username="invalid-username"))