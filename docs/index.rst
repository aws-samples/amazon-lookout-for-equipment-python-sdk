.. Lookout for Equipment SDK documentation master file, created by
   sphinx-quickstart on Mon May 24 10:20:31 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Lookout for Equipment SDK's documentation!
=====================================================

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started
   
   install

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Documentation
   
   userguide
   api

The **Amazon Lookout for Equipment SDK** is an open source Python package that
allows data scientists and software developers to easily build and deploy time
series anomalie detection models using Amazon Lookout for Equipment. This SDK
enables to do the following:

* Build dataset schema
* Data upload to the necessary S3 structure
* Train an anomaly detection model using Amazon Lookout for Equipment
* Build beautiful visualization for your model evaluation
* Configure and start an inference scheduler
* Manage schedulers (start, stop, delete) whenever necessary
* Visualize scheduler inferences results