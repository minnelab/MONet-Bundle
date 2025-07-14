from __future__ import annotations

import json
import os
import datetime
import jwt
import requests
from jwt import decode


def get_token(username: str, password: str) -> str:
    """
    Obtain an authentication token from the KTH IAM OpenID Connect service.

    Parameters
    ----------
    username : str
        The username for authentication.
    password : str
        The password for authentication.

    Returns
    -------
    str
        The authentication token as a string.

    Raises
    ------
    requests.HTTPError
        If the request to the authentication service fails.
    """
    url = "https://iam.cloud.cbh.kth.se/realms/cloud/protocol/openid-connect/token"
    data = {"grant_type": "password", "username": username, "password": password, "client_id": "monailabel-app"}

    response = requests.post(url, data=data)
    response.raise_for_status()
    return response.json()


def verify_valid_token_exists(username: str) -> bool:
    """
    Checks if a valid authentication token exists for the specified user.
    This function looks for an authentication token file in the user's home directory,
    loads the token, and verifies its validity by decoding the JWT token. It also prints
    the token's expiration time if available.

    Parameters
    ----------
    username : str
        The username for which to check the authentication token.

    Returns
    -------
    bool
        True if a valid token exists and is not expired, False otherwise.

    Raises
    ------
    None
    """
    home = os.path.expanduser("~")
    auth_path = os.path.join(home, ".monet", f"{username}_auth.json")

    if not os.path.exists(auth_path):
        return False

    with open(auth_path, "r") as token_file:
        token_data = json.load(token_file)

    token = token_data.get("access_token")

    if not token:
        return False

    try:
        decode(token, options={"verify_signature": False})
        print(f"Token for {username} is valid.")
        expiration = decode(token, options={"verify_signature": False}).get("exp")
        if expiration:
            expires_at = datetime.datetime.fromtimestamp(expiration, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            print(f"Token expires at: {expires_at}")
        # Check if token is expired
        if expiration and datetime.datetime.now(datetime.timezone.utc).timestamp() > expiration:
            print("Token is expired.")
            return False
        return True
    except jwt.ExpiredSignatureError:
        return False
    except jwt.DecodeError:
        return False


def welcome_message(token: str) -> str:
    """
    Generates a welcome message using the preferred username from a JWT token.
    Decodes the provided JWT token without verifying its signature, extracts the expiration time,
    prints when the token expires, and returns a welcome message including the preferred username.

    Parameters
    ----------
    token : str
        The JWT token as a string.

    Returns
    -------
    str
        A welcome message containing the preferred username from the token, or 'User' if not present.
    """
    decoded_token = decode(token, options={"verify_signature": False})
    expiration = decoded_token.get("exp")
    if expiration:
        expires_at = datetime.datetime.utcfromtimestamp(expiration).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"Token expires at: {expires_at}")

    return f"Welcome {decoded_token.get('preferred_username', 'User')}!"
