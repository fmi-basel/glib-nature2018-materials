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
        'numpy',
        'scipy',
        'pandas',
        'six',
        'statsmodels',
        'mahotas',
        'tqdm',
        'scikit-image >= 0.13.1',
        'opencv-python>=3.4',
        'future',
        'Pillow',
        # TODO Check versions with Urs on Lab machine.
        # TODO Is there something unnecessary?
        # TODO make dlutils a package and dependency or freeze the code and add as third party?
        #      if yes, we will need to add keras as extra dependency.
    ],
    extras_require={
        'gpu': ['tensorflow-gpu'],
        'cpu': ['tensorflow']
    },
)
