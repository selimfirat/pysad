from setuptools import setup, find_packages
import os

with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.rst'), encoding='utf-8') as f:
    readme = f.read()

with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'requirements.txt'),
          encoding='utf-8') as f:
    requirements = f.read().splitlines()

ver_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'pysad', 'version.py')
with open(ver_file) as f:
    exec(f.read())
VERSION = __version__

setup(
    name='pysad',
    version=VERSION,
    url='https://github.com/selimfirat/pysad',
    license='3-Clause BSD',
    author='Selim Firat Yilmaz',
    author_email='yilmazselimfirat@gmail.com',
    description='PySAD is an open-source python framework for anomaly detection on streaming multivariate data. ',
    long_description=readme,
    long_description_content_type='text/x-rst',
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    install_requires=requirements,
    setup_requires=['setuptools>=38.6.0'],
    download_url="https://github.com/selimfirat/pysad/archive/master.zip",
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Education',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology',
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "License :: OSI Approved :: BSD License",
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8"
)
