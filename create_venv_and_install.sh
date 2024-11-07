# Description: This file contains the function to create a virtual environment and install the requirements.
# The function will create a virtual environment with the name of the current directory and install the requirements from requirements.txt.
# It will also create an alias for the virtual environment and append it to the .bashrc file.
# The alias will also change the directory to the virtual environment directory and activate the virtual environment.

# Usage: create_venv_and_install
# Example: create_venv_and_install
# Dependencies: venv, pip, requirements.txt
# Output: Virtual environment created and requirements installed.
# Output: Alias created and appended to .bashrc.


function create_venv_and_install() {
    local venv_name=$(basename "$(pwd)")
    python -m venv "$venv_name"
    source "$venv_name/bin/activate"
    pip install -r requirements.txt

    # Extract the part of the venv name before the last "_" - e.g. venv name is "technical_assessment"
    local alias_name=${venv_name%_*}

    # Create an alias and append it to .bashrc
    echo "alias $alias_name='cd $CB/$venv_name && source $venv_name/bin/activate'" >> ~/.bashrc

    # Show the user the alias that was created
    echo "alias $alias_name created. Run 'source ~/.bashrc' to activate the alias."
}