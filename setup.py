from setuptools import setup, find_packages

setup(
    name='malta',
    version='0.3.1',
    description='A 2D model of atmospheric transport',
    author='Luke Western',
    license='MIT',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9, <3.11",
    install_requires=[
        "numpy==1.24.4",
        "pandas==2.0.3",
        "xarray==2023.10.1",
        "numba==0.58.1",
        "scipy>=1.11.4",
        "matplotlib>=3.8.0",
        "netCDF4>=1.7.2",
    ],
    include_package_data=True,
    zip_safe=False,
)

