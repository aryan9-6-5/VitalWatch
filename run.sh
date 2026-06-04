#!/bin/bash

USERNAME="aryan9-6-5"

echo "Fetching repositories for $USERNAME..."
echo

curl -s https://api.github.com/users/$USERNAME/repos | grep '"html_url"' | cut -d '"' -f4