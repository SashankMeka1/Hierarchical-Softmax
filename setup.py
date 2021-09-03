from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="torch_hier_softmax",
    description="Pytorch extension for hierarchical softmax NLP tasks for developers to use.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    version= "1.0.0",
    author="sashank meka",
    author_email="sashankmeka7@gmail.com",
    packages=['torch_hier_softmax'],
    include_package_data=True,
    python_requires=">=3.7.*",
    install_requires=['torch'],
    license="MIT License",
    zip_safe=False,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.7',
    ]
)
