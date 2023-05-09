from setuptools import setup, find_packages


with open('requirements.txt', 'r') as fd:
    required = [x.strip() for x in fd.readlines() if len(x.strip()) > 0]

setup(
    name='scientific-calibration',
    packages=find_packages(),
    install_required=required
)
