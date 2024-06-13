#!/bin/bash

# File containing the list of URLs
URL_FILE="urls.txt"

# Read each URL from the file and use curl to download the file
while IFS= read -r url; do
    curl -n -c ~/.urs_cookies -b ~/.urs_cookies -LJO --url "$url"
done < "$URL_FILE"
