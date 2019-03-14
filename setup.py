from setuptools import setup

authors = [
    'Urs Mayr',
    'Markus Rempfler',
]

setup(
    name='glib-nature2018-materials',
    version='0.1.0',
    author=authors,
    packages=['image_processing'],
    description='''Image Processing Package for the Paper
    " Self-organization and symmetry breaking in intestinal organoid development"''',
    long_description=open('README.md').read(),
    install_requires=[
        'numpy==1.14.5',
        'scipy==1.1.0',
        'pandas==0.23',
        'six>=1.11',
        'statsmodels==0.9',
        'mahotas==1.4.5',
        'tqdm>=4.26',
        'scikit-image>=0.14.1',
        'opencv-python>=3.4',
        'future>=0.16',
        'Pillow==5.2',
        'keras==2.0.8',
        'h5py>=2.8'
    ],
    extras_require={
        'gpu': ['tensorflow-gpu==1.10'],
        'cpu': ['tensorflow==1.10']
    },
)
