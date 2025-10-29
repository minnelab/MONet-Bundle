import unittest
import tempfile
import os
from MONet.auth import get_token, verify_valid_token_exists, welcome_message
from unittest.mock import MagicMock, patch
import json
import time
import shutil
import jwt
import requests
class TestMONetAuthHelpers(unittest.TestCase):
    
    
    def setUp(self):
        self.username = "test@maia.se"
        self.password = "my-secret-password!"
        self.token = "test-token"
        self.refresh_token = "test-refresh-token"
        
        self.temp_home_dir = tempfile.mkdtemp()
        os.environ["HOME"] = self.temp_home_dir
        os.makedirs(os.path.join(self.temp_home_dir, ".monet"), exist_ok=True)
        
        with open(os.path.join(self.temp_home_dir, ".monet", f"{self.username}_auth.json"), "w") as f:
            json.dump({"access_token": self.token, "refresh_token": self.refresh_token}, f)
            
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
        
        expiration_time = time.time() + 3600
        mock_decode.return_value = {"exp": expiration_time}
        self.assertTrue(verify_valid_token_exists(username=self.username))

    
    @patch("MONet.auth.decode")
    @patch("MONet.auth.requests.post")
    def test_verify_valid_token_exists_with_expired_token(self, mock_post, mock_decode):
        
        mock_post.return_value = MagicMock(status_code=200, json=lambda: {"access_token": self.token, "refresh_token": self.refresh_token})
        expiration_time = time.time() - 3600
        mock_decode.return_value = {"exp": expiration_time}
        self.assertTrue(verify_valid_token_exists(username=self.username))
        
        
    @patch("MONet.auth.decode")
    @patch("MONet.auth.requests.post")
    def test_verify_valid_token_exists_with_expired_token_and_request_exception(self, mock_post, mock_decode):
        
        mock_post.side_effect = requests.RequestException("Request exception")
        expiration_time = time.time() - 3600
        mock_decode.return_value = {"exp": expiration_time}
        self.assertFalse(verify_valid_token_exists(username=self.username))
    
    @patch("MONet.auth.json.load")
    @patch("MONet.auth.decode")
    def test_verify_valid_token_exists_with_expired_token_and_no_refresh_token(self, mock_decode, mock_load):
        mock_load.return_value = {"access_token": self.token, "refresh_token": None}
        expiration_time = time.time() - 3600
        mock_decode.return_value = {"exp": expiration_time}
        self.assertFalse(verify_valid_token_exists(username=self.username))
    
    @patch("MONet.auth.json.load")
    def test_verify_valid_token_exists_with_no_access_token_in_auth_file(self, mock_load):
        mock_load.return_value = {}
        self.assertFalse(verify_valid_token_exists(username=self.username))
    
    
    def test_verify_valid_token_exists_with_invalid_username(self):
        self.assertFalse(verify_valid_token_exists(username="invalid-username"))
    
    @patch("MONet.auth.os.path.exists")
    def test_verify_valid_token_exists_with_no_auth_file(self, mock_exists):
        mock_exists.return_value = False
        self.assertFalse(verify_valid_token_exists(username=self.username))

    @patch("MONet.auth.decode")
    def test_verify_valid_token_exists_with_expired_signature(self, mock_decode):
        mock_decode.side_effect = jwt.ExpiredSignatureError("Expired signature")
        self.assertFalse(verify_valid_token_exists(username=self.username))
    
    @patch("MONet.auth.decode")
    def test_verify_valid_token_exists_with_decode_error(self, mock_decode):
        mock_decode.side_effect = jwt.DecodeError("Decode error")
        self.assertFalse(verify_valid_token_exists(username=self.username))
    
    @patch("MONet.auth.decode")
    def test_welcome_message(self, mock_decode):
        mock_decode.return_value = {"exp": time.time() + 3600, "preferred_username": self.username}
        self.assertEqual(welcome_message(token=self.token), f"Welcome {self.username}!")