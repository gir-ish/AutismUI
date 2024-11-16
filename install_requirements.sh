#!/bin/bash

# Step 1: Verify if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found in the current directory."
    exit 1
fi

# Step 2: Backup the original requirements.txt
cp requirements.txt requirements_backup.txt
echo "Backup of requirements.txt created as requirements_backup.txt"

# Step 3: Filter problematic lines and create a cleaned version
echo "Cleaning problematic lines from requirements.txt..."
grep -v "/opt/conda/conda-bld/psutil_1656431268089/work" requirements.txt > cleaned_requirements.txt

# Step 4: Check if psutil is still needed
if ! grep -q "psutil" cleaned_requirements.txt; then
    echo "Adding 'psutil' to cleaned_requirements.txt with a default version..."
    echo "psutil==5.9.0" >> cleaned_requirements.txt
fi

# Step 5: Install dependencies
echo "Installing packages from cleaned_requirements.txt..."
pip install -r cleaned_requirements.txt

if [ $? -eq 0 ]; then
    echo "All packages installed successfully!"
else
    echo "There was an error installing some packages. Check the log for details."
fi
