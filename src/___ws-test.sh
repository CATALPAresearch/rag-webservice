#!/bin/bash

# Define the web service URL
URL="http://localhost:5000/process"

# Define the file to send
FILE_PATH="data/pdfs/1884/2022-Kurs1884-KE2.pdf"  # Change this to your actual file path

# Check if the file exists
if [[ ! -f "$FILE_PATH" ]]; then
  echo "Error: File $FILE_PATH not found!"
  exit 1
fi

# Send the file using curl
response=$(curl -X POST "$URL" \
     -F "file=@$FILE_PATH" \
     -F "user=12345")  # Replace 12345 with a test user ID

# Print response
echo "Server Response: $response"
