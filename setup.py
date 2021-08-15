from distutils.core import setup
from setuptools import find_packages

with open("README.rst", "r") as f:
  long_description = f.read()

setup(name='raytf',  # 包名
      # Reference from https://www.python.org/dev/peps/pep-0440/
      version='0.0.1',  # 版本号
      description='Tensorflow Cluster on Ray',
      long_description=long_description,
      author='junfan.zhang',
      author_email='junfan.zhang@outlook.com',
      url='https://github.com/zuston/raytf',
      install_requires=[],
      license='BSD License',
      packages=find_packages(),
      platforms=["all"],
      classifiers=[
          'Intended Audience :: Developers',
          'Operating System :: OS Independent',
          'Natural Language :: Chinese (Simplified)',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Topic :: Software Development :: Libraries'
      ],
      )