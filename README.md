# Amazon Lookout for Equipment Python SDK

[![Documentation Status](https://readthedocs.org/projects/amazon-lookout-for-equipment-sdk/badge/?version=latest)](https://amazon-lookout-for-equipment-sdk.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/lookoutequipment.svg)](https://badge.fury.io/py/lookoutequipment)

The Amazon Lookout for Equipment Python SDK is an open-source library that 
allows data scientists and software developers to easily build, train and deploy 
anomaly detection models for industrial time series data using 
[**Amazon Lookout for Equipment**](https://aws.amazon.com/lookout-for-equipment/)

The Amazon Lookout for Equipment Python SDK enables you to do the following:

- Build dataset schema
- Data upload to the necessary S3 structure
- Train an anomaly detection model using Amazon Lookout for Equipment
- Build beautiful visualization for your model evaluation
- Configure and start an inference scheduler
- Manage schedulers (start, stop, delete) whenever necessary
- Visualize scheduler inferences results


Getting Started With Sample Jupyter Notebooks
---------------------------------------------

The best way to quickly review how the Amazon Lookout for Equipment Python SDK 
works is to review the related example notebook.

This [notebook](examples/tutorial.ipynb) provides code and descriptions for creating 
and running a full project in Amazon Lookout for Equipment using the Amazon Lookout 
for Equipment Python SDK.


Example Notebooks in SageMaker
---------------------------------------------

In Amazon SageMaker, upload the Jupyter notebook from the **`examples/`** folder of this repository.

1. To run this example [Create a Notebook Instance](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-setup-working-env.html) in SageMaker.
2. Add an inline policy to your Amazon SageMaker role in IAM with the following JSON structure

```json
	{
	    "Version": "2012-10-17",
	    "Statement": [
	        {
	            "Effect": "Allow",
	            "Action": [
	                "lookoutequipment:*"
	            ],
	            "Resource": "*"
	        }
	    ]
	}
```
3. Upload the Jupyter notebook from **`examples/`** folder.
4. Run the notebook cells


Installing the Amazon Lookout for Equipment Python SDK
--------------------------------------------------

The Amazon Lookout for Equipment Python SDK is built to PyPI and can be 
installed with pip as follows:

```
	pip install lookoutequipment
```

You can install from source by cloning this repository and running a pip install
command in the root directory of the repository:

```
	git clone https://github.com/aws-samples/amazon-lookout-for-equipment-python-sdk.git
	cd amazon-lookout-for-equipment-python-sdk
	pip install .
```


Supported Operating Systems
---------------------------------------------

The Amazon Lookout for Equipment Python SDK supports Unix/Linux and Mac.


Supported Python Versions
---------------------------------------------

The Amazon Lookout for Equipment Python SDK is tested on:

* Python 3.6


Overview of SDK
---------------

The Amazon Lookout for Equipment Python SDK provides a Python API that enables 
you to easily build, train and deploy anomaly detection models for industrial 
time series data using Amazon Lookout for Equipment, and directly in your python 
code and Jupyter notebooks.

Using this SDK you can:

1. Build dataset schema
2. Data upload to the necessary S3 structure
3. Train an anomaly detection model using Amazon Lookout for Equipment
4. Build beautiful visualization for your model evaluation

![Visualization example](docs/images/model_evaluation.png)

5. Configure and start an inference scheduler
6. Manage schedulers (start, stop, delete) whenever necessary
7. Visualize scheduler inferences results

For a detailed API reference of the Amazon Lookout for Equipment Python SDK,
be sure to view its documentation hosted on [**readthedocs**](https://amazon-lookout-for-equipment-sdk.readthedocs.io/en/latest/).


Amazon Lookout for Equipment
---------------------------------------------

Amazon Lookout for Equipment uses the data from your sensors to detect abnormal 
equipment behavior, so you can take action before machine failures occur and 
avoid unplanned downtime.


AWS Permissions
---------------
As a managed service, Amazon Lookout for Equipment performs operations on your 
behalf on AWS hardware that is managed by Amazon Lookout for Equipment. Amazon
Lookout for Equipment can perform only operations that the user permits.  You 
can read more about which permissions are necessary in the 
[**AWS Documentation**](https://docs.aws.amazon.com/lookout-for-equipment/latest/ug/what-is.html).

The Amazon Lookout for Equipment Python SDK should not require any additional 
permissions aside from what is required for using `boto3`  However, if you are
using an IAM role with a path in it, you should grant permission for `iam:GetRole`.


Security
---------------

See [**CONTRIBUTING.md**](CONTRIBUTING.md) for more information.


Licensing
---------
Amazon Lookout for Equipment Python SDK is licensed under the Apache 2.0 
License. It is copyright 2021 Amazon.com, Inc. or its affiliates. All Rights 
Reserved. The license is available at: **http://aws.amazon.com/apache2.0/**.