from setuptools import setup, find_packages

#with open("README.md", "r") as fh:
#    long_description = fh.read()

version = '5.1.0'
packages = ['perex','perex.aux']

setup(name='perex',
      version=version,
      packages=packages,
      description='Python Environment for Relating Electrochemistry and XAS data',
      author='Lucía Pérez Ramírez',
      author_email='lucia.perez@synchrotron-soleil.fr',
      #long_description=long_description,
      license='GPL >= v.3',
      platforms=['Linux', 'Windows'],
      url="https://github.com/GhostDeini/perex",
      python_requires='>=3.9',
      #package_data=package_data,
      #scripts=scripts
      )
