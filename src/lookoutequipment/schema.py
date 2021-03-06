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

import json
import os
import pandas as pd
import s3fs

from typing import List, Dict


def create_data_schema_from_dir(root_dir):
    """
    Generates a data schema compatible for Lookout for Equipment from a local 
    directory
    
    Parameters:
        root_dir (string):
            a path pointing to the root directory where all the CSV files are 
            located
            
    Returns:
        string:
            a JSON-formatted string ready to be used as a schema for a Lookout
            for Equipment dataset
    """
    # List of the directories from the training data 
    # directory: each directory corresponds to a subsystem:
    components = []
    for _, dirs, _ in os.walk(root_dir):
        for subsystem in dirs:
            if subsystem != '.ipynb_checkpoints':
                components.append(subsystem)

    # Loops through each subdirectory found in the root dir:
    DATASET_COMPONENT_FIELDS_MAP = dict()
    for subsystem in components:
        subsystem_tags = ['Timestamp']
        for root, _, files in os.walk(f'{root_dir}/{subsystem}'):
            for file in files:
                fname = os.path.join(root, file)
                current_subsystem_df = pd.read_csv(fname, nrows=1)
                subsystem_tags = subsystem_tags + current_subsystem_df.columns.tolist()[1:]

            DATASET_COMPONENT_FIELDS_MAP.update({subsystem: subsystem_tags})
            
    schema = create_data_schema(DATASET_COMPONENT_FIELDS_MAP)
    
    return schema
    
def create_data_schema_from_s3_path(s3_path):
    """
    Generates a data schema compatible for Lookout for Equipment from an S3 
    directory
    
    Parameters:
        s3_path (string):
            a path pointing to the root directory on S3 where all the CSV files 
            are located
            
    Returns:
        string:
            a JSON-formatted string ready to be used as a schema for a Lookout
            for Equipment dataset
    """
    # We should have only directories at the first level of this S3 path:
    fs = s3fs.S3FileSystem()
    components = fs.ls(s3_path)

    # Loops through each subdirectory found in the root dir:
    DATASET_COMPONENT_FIELDS_MAP = dict()
    for subsystem in components:
        # The first tag should always be Timestamp
        subsystem_tags = ['Timestamp']
        files = fs.ls(subsystem)

        # Open each file to get their list of fields:
        for file in files:
            current_subsystem_df = pd.read_csv(f's3://{file}', nrows=1)
            subsystem_tags = subsystem_tags + current_subsystem_df.columns.tolist()[1:]

            DATASET_COMPONENT_FIELDS_MAP.update({subsystem.split('/')[-1]: subsystem_tags})
            
    # Generate the associated JSON schema:
    schema = create_data_schema(DATASET_COMPONENT_FIELDS_MAP)
    
    return schema

def create_data_schema(component_fields_map: Dict):
    """
    Generates a JSON formatted string from a dictionary
    
    Parameters:
        component_fields_map (dict):
            a dictionary containing a field maps for the dataset schema
            
    Returns:
        string:
            a JSON-formatted string ready to be used as a schema for a dataset
    """
    schema = json.dumps(
        _create_data_schema_map(
            component_fields_map=component_fields_map
        )
    )
    
    return schema

def _create_data_schema_map(component_fields_map: Dict):
    """
    Generate a dictionary with the JSON format expected by Lookout for Equipment
    to be used as the schema for a dataset at ingestion, training time and
    inference time
    
    Parameters:
        component_fields_map (dict):
            a dictionary containing a field maps for the dataset schema

    Returns:
        dict:
            a dictionnary containing the detailed schema built from the original
            dictionary mapping
    """
    # Build the schema for the current component:
    component_schema_list = [_create_component_schema(
            component_name, 
            component_fields_map[component_name]
        ) for component_name in component_fields_map
    ]
    
    # The root of the schema is a "Components" tag:
    data_schema = dict()
    data_schema['Components'] = component_schema_list

    return data_schema

def _create_component_schema(component_name: str, field_names: List):
    """
    Build a schema for a given component and fieds list
    
    Parameters
        component_name (string):
            name of the component to build a schema for
        
        field_names (list of strings):
            name of all the fields included in this component
            
    Returns:
        dict:
            A dictionnary containing the detailed schema for a given component
    """
    # Test if the field names is correct for this component:
    if len(field_names) == 0:
        raise Exception(f'Field names for component {component_name} should not be empty')
    if len(field_names) == 1:
        raise Exception(f'Component {component_name} must have at least one sensor beyond the timestamp')
    
    # The first field is a timestamp:
    col_list  = [{'Name': field_names[0], 'Type': 'DATETIME'}]
    
    # All the others are float values:
    col_list = col_list + [
        {'Name': field_name, 'Type': 'DOUBLE'} 
        for field_name in field_names[1:]
    ]
    
    # Build the schema for this component:
    component_schema = dict()
    component_schema['ComponentName'] = component_name
    component_schema['Columns'] = col_list
            
    return component_schema