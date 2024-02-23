#!/bin/bash

# Add all changes
git add .

# Commit with a timestamp
git commit -m "Auto update results at $(date)"

# Push to remote repository
git push origin main