import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="scimmunity-racng", # Replace with your own username
    version="0.0.1",
    author="Rachel Ng",
    author_email="rachelng323@gmail.com",
    description="Single cell RNAseq analysis.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vdjonsson/scimmunity",
    packages=['scimmunity'],
    package_dir={'scimmunity':'scimmunity'}, 
    package_data={'scimmunity': ['data/markersets/*']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)