"""Setup module for project."""

from setuptools import setup, find_packages

setup(
        name='mp18-motion-prediction-code',
        version='0.1',
        description='Skeleton code for Machine Perception Human Motion Prediction project.',

        author='Manuel Kaufmann',
        author_email='manuel.kaufmann@inf.ethz.ch',

        packages=find_packages(exclude=[]),
        python_requires='>=3.5',
        install_requires=[
            'numpy',
            'matplotlib',
            'pandas'

            # Install the most appropriate version of Tensorflow
            # Ref. https://www.tensorflow.org/install/
            # 'tensorflow',
        ],
)
