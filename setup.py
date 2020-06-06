from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='FContin',
      version='0.1',
      description='Numerical continuation using just the function',
      url='https://github.com/gpavanb1/FContin',
      author='gpavanb1',
      author_email='gpavanb@gmail.com',
      license='MIT',
      packages=['fcontin'],
      install_requires=["numpy",
      "scipy",
      "jax",
      "jaxlib",
      "numdifftools",
      "pacopy"],
      long_description=long_description,
      long_description_content_type="text/markdown",
      zip_safe=False)