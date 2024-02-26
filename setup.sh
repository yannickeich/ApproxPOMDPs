#!/bin/bash
#check version
version=$(python -c "import platform; print(platform.python_version())")
if [ "$version" != "3.9.10" ]
then
	echo "Requires Python 3.9.10"
	exit 1
fi

#upgrade pip
pip install --upgrade pip
#install packages
pip install -r requirements.txt

#add the system path to the python site-packages
CURRENT_PATH=$(pwd)
VENV_PYTHON_PATH=$(dirname $(which python))
cd "$VENV_PYTHON_PATH/../lib/python3.9/site-packages/"
echo $CURRENT_PATH > assumed_pomdp.pth