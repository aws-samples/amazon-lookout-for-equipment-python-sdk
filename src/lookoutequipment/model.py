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

import boto3
import json
import pandas as pd
import time

from datetime import datetime
from .schema import create_data_schema

def list_models(
    model_name_prefix=None, 
    dataset_name_prefix=None,
    max_results=50
):
    """
    List all the models available in the current account
    
    Parameters:
        model_name_prefix (string):
            Prefix to filter on the model name to look for (default: None)
            
        dataset_name_prefix (string):
            Prefix to filter the dataset name: if used, only models
            making use of this particular dataset are returned (default: None)

        max_results (integer):
            Max number of datasets to return (default: 50)
            
    Returns:
        List of strings:
            List of all the models corresponding to the input parameters
            (regions and dataset)
    """
    # Initialization:
    models_list = []
    has_more_records = True
    client = boto3.client('lookoutequipment')
    
    # Building the request:
    list_models_request = {"MaxResults": max_results}
    if model_name_prefix is not None:
        list_models_request["ModelNameBeginsWith"] = model_name_prefix
    if dataset_name_prefix is not None:
        list_models_request["DatasetNameBeginsWith"] = dataset_name_prefix

    # We query for the list of models, until there are none left to fetch:
    while has_more_records:
        # Query for the list of L4E models available for this AWS account:
        list_models_response = client.list_models(**list_models_request)
        if "NextToken" in list_models_response:
            list_models_request["NextToken"] = list_models_response["NextToken"]
        else:
            has_more_records = False

        # Add the model names to the list:
        model_summaries = list_models_response["ModelSummaries"]
        for model_summary in model_summaries:
            models_list.append(model_summary['ModelName'])

    return models_list

class LookoutEquipmentModel:
    """
    A class to manage Lookout for Equipment models
    
    Attributes:
        dataset_name (string):
            The name of the dataset used to train the model attached to a
            given class instance
        
        model_name (string):
            The name of the model attached to a given class instance
            
        create_model_request (dict):
            The parameters to be used to train the model
    """
    def __init__(self, model_name, dataset_name):
        """
        Create a new instance to configure all the attributes necessary to 
        manage a Lookout for Equipment model.
        
        Parameters:
            model_name (string):
                the name of the model to manage
            dataset_name (string):
                the name of the dataset associated to the model
        """
        self.client = boto3.client('lookoutequipment')
        self.model_name = model_name
        self.create_model_request = dict()
        self.create_model_request.update({
            'ModelName': model_name,
            'DatasetName': dataset_name
        })
        
    def set_off_conditions(self, off_conditions_string):
        """
        Tells Lookout for Equipment to use one of the signals as a guide to
        tell if the asset/process is currently on or off.
        
        Parameters:
            off_conditions_string (string):
                A string with the format `component_name\\tag_name>0.0` where
                the condition can either be `<` or `>` with a real value
                materializing the boundary used to identify off time from on
                time.
        """
        self.create_model_request.update({
            'OffCondition': off_conditions_string
        })
        
    def set_label_data(self, bucket, prefix, access_role_arn):
        """
        Tell Lookout for Equipment to look for labelled data to train the
        model and where to find them on S3
        
        Parameters:
            bucket (string):
                Bucket name where the labelled data can be found
                
            prefix (string):
                Prefix where the labelled data can be found
                
            access_role_arn (string):
                A role that Lookout for Equipment can use to access the bucket
                and prefix aforementioned
        """
        labels_input_config = dict()
        labels_input_config['S3InputConfiguration'] = dict([
            ('Bucket', bucket),
            ('Prefix', prefix)
        ])
        self.create_model_request.update({
            'RoleArn': access_role_arn,
            'LabelsInputConfiguration': labels_input_config
        })
        
    def set_target_sampling_rate(self, sampling_rate):
        """
        Set the sampling rate to use before training the model
        
        Parameters:
            sampling_rate (string):
                One of [PT1M, PT5S, PT15M, PT1S, PT10M, PT15S, PT30M, PT10S, 
                PT30S, PT1H, PT5M]
        """
        self.create_model_request.update({
            'DataPreProcessingConfiguration': {
                'TargetSamplingRate': sampling_rate
            },
        })
    
    def set_time_periods(self, 
                         evaluation_start, 
                         evaluation_end, 
                         training_start, 
                         training_end):
        """
        Set the training / evaluation time split
        
        Parameters:
            evaluation_start (datetime):
                Start of the evaluation period

            evaluation_end (datetime):
                End of the evaluation period

            training_start (datetime):
                Start of the training period

            training_end (datetime):
                End of the training period
        """
        self.create_model_request.update({
            'TrainingDataStartTime': training_start.to_pydatetime(),
            'TrainingDataEndTime': training_end.to_pydatetime(),
            'EvaluationDataStartTime': evaluation_start.to_pydatetime(),
            'EvaluationDataEndTime': evaluation_end.to_pydatetime()
        })
        
    def set_off_condition(self,
                         off_condition):
        """
        Configure off-time detection using one of your machine’s sensors.
        
        Parameters:
            off_condition (string):
                Sensor representative of the machine’s on/off state.
                Ex: 'tag_name < 1000'

        """
        
        self.create_model_request.update({
            'OffCondition': off_condition
        })
        
    def set_subset_schema(self, field_map):
        """
        Configure the inline data schema that will let Lookout for Equipment
        knows that it needs to select a subset of all the signals configured
        at ingestion
        
        Parameters:
            field_map: string
                A JSON string describing which signals to keep for this model
        """
        data_schema_for_model = {
            'InlineDataSchema': create_data_schema(field_map),
        }
        self.create_model_request['DatasetSchema'] = data_schema_for_model

    def train(self):
        """
        Train the model as configured with this object
        
        Returns:
            string:
                The create model API response in JSON format
        """
        try:
            create_model_response = self.client.create_model(
                **self.create_model_request
            )
            return create_model_response
            
        except self.client.exceptions.ConflictException as e:
            print((
                f'A model with this name ({self.create_model_request["ModelName"]}) '
                'already exists or is currently in use for another operation. '
                'Change the name and try again'
            ))
            return False
            
        except Exception as e:
            raise(e)
        
    def delete(self):
        """
        Delete the current model

        Returns:
            string:
                The delete model API response in JSON format
        """
        try:
            delete_model_response = self.client.delete_model(
                ModelName=self.model_name
            )
            return delete_model_response
            
        except self.client.exceptions.ConflictException as e:
            print(('Model is currently being used (a training might be in '
                  'progress. Wait for the process to be completed and '
                  'retry.'))
            return False
            
        except Exception as e:
            raise(e)

    def poll_model_training(self, sleep_time=60):
        """
        This function polls the model describe API and prints a status until the
        training is done.
        
        Parameters:
            sleep_time (integer)
                How many seconds should we wait before polling again
                (default: 60)
        """
        describe_model_response = self.client.describe_model(
            ModelName=self.model_name
        )
        
        status = describe_model_response['Status']
        while status == 'IN_PROGRESS':
            time.sleep(sleep_time)
            describe_model_response = self.client.describe_model(
                ModelName=self.model_name
            )
            status = describe_model_response['Status']
            print(
                str(pd.to_datetime(datetime.now()))[:19],
                "| Model training:", 
                status
            )