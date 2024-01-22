from setuptools import find_packages,setup 
from typing import List



def get_requirment(filepath:str)->List[str]:

    requirment = []
    HYPHEN_E = '-e .'

    with open(filepath)as fil_obj:
        requirment = fil_obj.readlines()
        requirment=[req.replace('\n','') for req in requirment]
    if HYPHEN_E in requirment:
        requirment.remove(HYPHEN_E) 
    
    return requirment



setup(
name='Full Ml Project',
version='0.1',
author='Akram',
packages=find_packages(),
install_requires = get_requirment('requirments.txt')    
)