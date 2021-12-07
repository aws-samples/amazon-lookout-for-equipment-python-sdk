.. _api:

=================
API Documentation
=================

Full API documentation of the *lookoutequipment* Python package.

Schema
======

.. automodule:: src.lookoutequipment.schema
    :no-members:
    :no-inherited-members:

.. currentmodule:: src.lookoutequipment

.. autosummary::
   :toctree: generated/
   :template: function.rst

   schema.create_data_schema_from_dir
   schema.create_data_schema_from_s3_path
   schema.create_data_schema

Datasets
========

.. automodule:: src.lookoutequipment.dataset
    :no-members:
    :no-inherited-members:

.. currentmodule:: src.lookoutequipment

.. autosummary::
   :toctree: generated/
   :template: function.rst

   dataset.list_datasets
   dataset.load_dataset
   dataset.upload_dataset
   dataset.prepare_inference_data
   dataset.generate_replay_data
   
   :template: class.rst
   
   dataset.LookoutEquipmentDataset
   
Models
======

.. automodule:: src.lookoutequipment.model
    :no-members:
    :no-inherited-members:

.. currentmodule:: src.lookoutequipment

.. autosummary::
   :toctree: generated/
   :template: function.rst
   
   model.list_models
   
   :template: class.rst
   
   model.LookoutEquipmentModel
   
Evaluation
==========

.. automodule:: src.lookoutequipment.evaluation
    :no-members:
    :no-inherited-members:
    
.. currentmodule:: src.lookoutequipment

.. autosummary::
   :toctree: generated/
   :template: class.rst
   
   evaluation.LookoutEquipmentAnalysis
   
Scheduler
=========

.. automodule:: src.lookoutequipment.scheduler
    :no-members:
    :no-inherited-members:
    
.. currentmodule:: src.lookoutequipment

.. autosummary::
   :toctree: generated/
   :template: class.rst
   
   scheduler.LookoutEquipmentScheduler
   scheduler.LookoutEquipmentSchedulerInspector
   
Plot
====

.. automodule:: src.lookoutequipment.plot
    :no-members:
    :no-inherited-members:
    
.. currentmodule:: src.lookoutequipment

.. autosummary::
   :toctree: generated/
   :template: function.rst

   plot.plot_histogram_comparison
   plot.plot_event_barh

   :template: class.rst
   
   plot.TimeSeriesVisualization