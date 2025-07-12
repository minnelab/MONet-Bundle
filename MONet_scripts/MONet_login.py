#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import json
import os

import requests

from MONet.auth import get_token, verify_valid_token_exists, welcome_message
from MONet.utils import get_available_models


def get_arg_parser():
    parser = argparse.ArgumentParser(description="MAIA Segmentation Portal Login Script")
    parser.add_argument("--username", type=str, required=True, help="Username for MAIA Segmentation Portal")
    parser.add_argument("--password", type=str, required=False, help="Password for MAIA Segmentation Portal")
    parser.add_argument("--list-models", action="store_true", help="List available models for segmentation")

    return parser


def main():
    parser = get_arg_parser()
    args = parser.parse_args()
    if not args.password:
        if not verify_valid_token_exists(args.username):
            args.password = getpass.getpass(prompt="Password for MAIA Segmentation Portal: ")

    print("")
    print("         WELCOME TO ")
    print("")

    logo_lines = [
        r"    ███╗   ███╗ █████╗ ██╗ █████╗                                                                       ",
        r"    ████╗ ████║██╔══██╗██║██╔══██╗                                                                      ",
        r"    ██╔████╔██║███████║██║███████║                                                                      ",
        r"    ██║╚██╔╝██║██╔══██║██║██╔══██║                                                                      ",
        r"    ██║ ╚═╝ ██║██║  ██║██║██║  ██║                                                                      ",
        r"    ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝                                                                      ",
        r"                                                                                                        ",
        r"    ███████╗███████╗ ██████╗ ███╗   ███╗███████╗███╗   ██╗████████╗ █████╗ ████████╗██╗ ██████╗ ███╗   ██╗",
        r"    ██╔════╝██╔════╝██╔════╝ ████╗ ████║██╔════╝████╗  ██║╚══██╔══╝██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║",
        r"    ███████╗█████╗  ██║  ███╗██╔████╔██║█████╗  ██╔██╗ ██║   ██║   ███████║   ██║   ██║██║   ██║██╔██╗ ██║",
        r"    ╚════██║██╔══╝  ██║   ██║██║╚██╔╝██║██╔══╝  ██║╚██╗██║   ██║   ██╔══██║   ██║   ██║██║   ██║██║╚██╗██║",
        r"    ███████║███████╗╚██████╔╝██║ ╚═╝ ██║███████╗██║ ╚████║   ██║   ██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║",
        r"    ╚══════╝╚══════╝ ╚═════╝ ╚═╝     ╚═╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝",
        r"                                                                                                        ",
    ]
    for line in logo_lines:
        print(line)

    if verify_valid_token_exists(args.username):
        print("\n")
        print(f"Welcome back {args.username}!")
        if args.list_models:
            home = os.path.expanduser("~")
            auth_path = os.path.join(home, ".monet", f"{args.username}_auth.json")
            with open(auth_path, "r") as token_file:
                token_data = json.load(token_file)
                token = token_data.get("access_token")
                if not token:
                    print("Access token not found. Please log in again.")
                    return
            models = get_available_models(token, args.username)
            print("\n")
            print("Available models for segmentation:")
            for model in models.keys():
                print(f"- {model}")
            return
        return

    # Simulate login process
    if args.username and args.password:
        print("\n")
        print(f"Logging in as {args.username}...")
        try:
            response = get_token(args.username, args.password)
            token = response.get("access_token")
            home = os.path.expanduser("~")
            auth_path = os.path.join(home, ".monet", f"{args.username}_auth.json")
            os.makedirs(os.path.dirname(auth_path), exist_ok=True)
            with open(auth_path, "w") as token_file:
                json.dump(response, token_file)
            print("\n")
            print(welcome_message(token))
            if args.list_models:
                home = os.path.expanduser("~")
                auth_path = os.path.join(home, ".monet", f"{args.username}_auth.json")
                with open(auth_path, "r") as token_file:
                    token_data = json.load(token_file)
                    token = token_data.get("access_token")
                    if not token:
                        print("Access token not found. Please log in again.")
                        return
                models = get_available_models(token, args.username)
                print("\n")
                print("Available models for segmentation:")
                for model in models.keys():
                    print(f"- {model}")
                return

        except requests.HTTPError as e:
            print(f"Login failed: {e.response.text}")
    else:
        print("Username and password are required.")


if __name__ == "__main__":
    main()
