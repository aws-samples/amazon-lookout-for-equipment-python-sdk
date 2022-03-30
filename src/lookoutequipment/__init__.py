#   Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
  
#   Licensed under the Apache License, Version 2.0 (the "License").
#   You may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
  
#       http://www.apache.org/licenses/LICENSE-2.0
  
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""
The :mod:`lookoutequipment` module includes methods that offers a higher level
API (a.k.a SDK) to deal with the constructs from the Amazon Lookout for 
Equipment service
"""

from .dataset import *
from .evaluation import *
from .model import *
from .plot import *
from .scheduler import *
from .schema import *

__version__ = '0.1.4'
__all__ = [
    'list_datasets',
    'load_dataset',
    'upload_dataset',
    'upload_file_to_s3',
    'prepare_inference_data',
    'generate_replay_data',
    'LookoutEquipmentDataset',
    
    'LookoutEquipmentAnalysis',
    
    'list_models',
    'LookoutEquipmentModel',
    
    'TimeSeriesVisualization',
    'compute_bin_edges',
    'plot_histogram_comparison',
    'plot_event_barh',
    'plot_range',
    
    'LookoutEquipmentSchedulerInspector',
    'LookoutEquipmentScheduler',
    
    'create_data_schema_from_dir',
    'create_data_schema_from_s3_path',
    'create_data_schema'
]