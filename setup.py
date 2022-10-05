from setuptools import setup

#with open("README.md", "r") as fh:
#    long_description = fh.read()

version = '1.0.0'
packages = ['perex']

setup(name='perex',
      version=version,
      description='Python Environment for Relating Electrochemistry and XAS data',
      author='Lucía Pérez Ramírez',
      author_email='lucia.perez@synchrotron-soleil.fr',
      #long_description=long_description,
      license='GPL >= v.3',
      platforms=['Linux', 'Windows'],
      url="https://github.com/GhostDeini/perex",
      packages=packages,
      python_requires='>=3.9',
      #package_data=package_data,
      #scripts=scripts
      )
