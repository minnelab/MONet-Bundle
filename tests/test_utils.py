import unittest
import tempfile
import os
from MONet.utils import get_available_models
from unittest.mock import MagicMock, patch
import json
import time
import shutil
import requests

class TestMONetUtilsHelpers(unittest.TestCase):
    
    
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

    @patch("MONet.utils.requests.post")
    def test_get_available_models(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200, json=lambda: {"models": {"test-model-segmentation": {"monai_label": "https://test-model-segmentation-portal.com"}}})
        models = get_available_models(token=self.token, username=self.username)
        self.assertEqual(models, {"test-model": "https://test-model-segmentation-portal.com"})
        mock_post.assert_called_once_with(
            "https://maia.app.cloud.cbh.kth.se/maia/maia-segmentation-portal/models/",
            data={"id_token": self.token, "username": self.username}
        )
        
    
    @patch("MONet.utils.requests.post")
    def test_get_available_models_with_request_exception(self, mock_post):
        mock_post.side_effect = requests.RequestException("Request exception")
        models = get_available_models(token=self.token, username=self.username)
        self.assertEqual(models, {})
        mock_post.assert_called_once_with(
            "https://maia.app.cloud.cbh.kth.se/maia/maia-segmentation-portal/models/",
            data={"id_token": self.token, "username": self.username}
        )
    