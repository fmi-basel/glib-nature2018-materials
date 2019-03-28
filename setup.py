import os
import platform
from setuptools import setup, find_packages
import shutil

if platform.system() == "Windows":
    mod_spatialite_dir = 'mod_spatialite-wn'
elif platform.system() == "Linux":
    mod_spatialite_dir = 'mod_spatialite-wn'
else:
    print(
        "NOTE: mod_spatialite-4.3.0a will not be installed via setuptools."
        "Please manually compile 'mod_spatialite' and copy the extension and "
        "dependencies to <install_dir>/mod_spatialite/."
        "Please refer to https://www.gaia-gis.it/gaia-sins/ for further "
        "information."
    )
    mod_spatialite_dir = None

setup(
    name='experiment-organizer',
    version='0.1.0',
    author='D. Vischi, U. Mayr',
    author_email='dario.vischi@fmi.ch, urs.mayr@fmi.ch',
    packages=find_packages(exclude=['tests']),
    package_dir={'experiment_organizer': './experiment_organizer'},
    # read additional package data from MANIFEST.in
    # not compatible with package_data
    include_package_data=False,
    # install files which are within the package,
    # but e.g. not within a module (a folder including an __init__.py file)
    package_data={
        'experiment_organizer': ["%s/*" % mod_spatialite_dir or '.']
    },
    data_files=[],
    description='Organizer for experiments of the Liberali group at the FMI.',
    long_description=open('README.md').read(),
    url='https://github.com/fmi-basel/glib-nature2018-materials/tree/organoid_linking_algorithm',
    license='LICENSE.txt',
    dependency_links=[
        'https://github.com/geoalchemy/geoalchemy2/tarball/0.5.0'
    ],
    install_requires=[
        "numpy >= 1.15.1",
        "pandas >= 0.23.4",
        "Pillow >= 5.2.0",
        "shapely >= 1.6.4",
        "sqlalchemy >= 1.2.11",
        "xlrd >= 1.1.0",
        "scipy >= 1.1.0",
        "scikit-image >= 0.14.0"
    ],
    test_suite='tests',
)
