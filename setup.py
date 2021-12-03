from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()
    
CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: Apache Software License',
    'Programming Language :: Python',
    'Topic :: Software Development',
    'Topic :: Scientific/Engineering',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8'
]
    
INSTALL_REQUIRES = open('requirements.txt','r').readlines()
    
setup(
    name='lookoutequipment',
    version='0.1.0',
    author='Michael Hoarau',
    author_email='michoara@amazon.com',
    description='Python SDK for Amazon Lookout for Equipment',
    license='Apache-2.0',
    url='https://github.com/awslabs/amazon-lookout-for-equipment-python-sdk',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(where='src', include=['lookoutequipment']),
    package_dir={"": "src"},
    zip_safe=False,
    install_requires=INSTALL_REQUIRES
)
