from setuptools import setup

setup(
    name='relax',
    version='0.1.0',
    description='RELAX: a framework for explainability in representation learning',
    url='https://github.com/Wickstrom/RELAX',
    author='Kristoffer Wickstrom',
    author_email='kwi030@uit.no',
    license='MIT',
    packages=['pyexample'],
    install_requires=['torch',
                      'torchvision',
                      ],

    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Programming Language :: Python :: 3',
    ],
) 