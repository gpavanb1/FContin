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
      classifiers=[
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3 :: Only',
      ],
      keywords='arclength python continuation numerical',
      project_urls={  # Optional
        'Bug Reports': 'https://github.com/gpavanb1/FContin/issues',
        'Source': 'https://github.com/gpavanb1/FContin/',
      },
      zip_safe=False)