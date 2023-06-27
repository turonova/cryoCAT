from setuptools import setup

setup(
    name='cryoCAT',
    version='0.1.0',    
    description='Contextual Analysis Tools for CryoET',
    url='https://github.com/turonova/cryoCAT',
    author='Beata Turonova',
    author_email='beata.turonova@gmail.com',
    license='GPLv3+',
    packages=['cryocat'],
    install_requires=['scipy',
                      'numpy',
                      'pandas',
                      'scikit-image',
                      'starfile',
                      'emfile',
                      'mrcfile',
                      'matplotlib',
                      'seaborn',
                      'scikit-learn',
                      'einops'
                      ],

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
)
