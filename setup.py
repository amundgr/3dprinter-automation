#!/usr/bin/env python

from distutils.core import setup

setup(name='automation_tools',
      version='1.0',
      description='Python Distribution Utilities',
      author='Amund Grorud',
      author_email='amund@grorud.net',
      packages=["automation_tools"],
      install_requires=[
          "RPi.GPIO",
          "numpy",
          "matplotlib",
          "scipy",
          "opencv-python",
          "pyserial",
      ],
     )