'''from typing import List
from setuptools import find_packages,setup
#from typing_1 import List

HYPEN_E_DOT= '-e.'

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
        return requirements

setup(
    name="Wafer Fault Dectection",
    version="0.0.1",
    author='Mohit Singh',
    author_email= 'mohitsingh2@yahoo.com',
    install_requires= get_requirements('requirements.txt'),
    packages=find_packages()
)'''

from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    author="Mohit Singh",
    author_email="mohitsingh2@yahoo.com",
    packages=find_packages(),
    install_requires=[]
)