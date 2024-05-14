import setuptools
import re


def get_property(prop, project):
    # https://stackoverflow.com/questions/17791481/creating-a-version-attribute-for-python-packages-without-getting-into-troubl/41110107
    result = re.search(
        r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop),
        open(project + "/__init__.py").read(),
    )
    return result.group(1)


with open("README.md", "r") as fh:
    long_description = fh.read()

project_name = "lightquakevisualizer"
setuptools.setup(
    name=project_name,
    version=get_property("__version__", project_name),
    author="Thomas Ulrich",
    description="A collection of scripts to visualize SeisSol output using pyvista",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Thomas-Ulrich/light-quake-visualizer",
    packages=setuptools.find_packages(),
    entry_points={
        "console_scripts": [
            "light_quake_visualizer = lightquakevisualizer.lightQuakeVisualizer:main",
            "generate_color_bar = lightquakevisualizer.generateColorBar:main",
            "image_combiner = lightquakevisualizer.imageCombiner:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "cmasher",
        "cmcrameri",
        "matplotlib",
        "numpy",
        "pyvista",
        "seissolxdmf>=0.1.3",
        "vtk",
        "Pillow",
    ],
)
