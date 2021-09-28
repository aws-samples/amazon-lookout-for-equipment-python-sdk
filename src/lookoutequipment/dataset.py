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
import numpy as np
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import time
import urllib.request
import zipfile

from botocore.exceptions import ClientError
from datetime import datetime, timedelta
from tqdm import tqdm

from .model import list_models
from .schema import create_data_schema, create_data_schema_from_dir, create_data_schema_from_s3_path


def list_datasets(dataset_name_prefix=None, max_results=50):
    """
    List all the Lookout for Equipment datasets available in this account.
    
    Parameters:
        dataset_name_prefix (string):
            prefix to filter out all the datasets which names starts by 
            this prefix. Defaults to None to list all datasets.
            
        max_results (integer):
            Max number of datasets to return (default: 50)
            
    Returns:
        list of strings:
            A list with all the dataset names found in the current region
    """
    # Initialization:
    dataset_list = []
    has_more_records = True
    client = boto3.client('lookoutequipment')
    
    # Building the request:
    kargs = {"MaxResults": max_results}
    if dataset_name_prefix is not None:
        kargs["DatasetNameBeginsWith"] = dataset_name_prefix
    
    # We query for the list of datasets, until there are none left to fetch:
    while has_more_records:
        # Query for the list of L4E datasets available for this AWS account:
        list_datasets_response = client.list_datasets(**kargs)
        if "NextToken" in list_datasets_response:
            kargs["NextToken"] = list_datasets_response["NextToken"]
        else:
            has_more_records = False
        
        # Add the dataset names to the list:
        dataset_summaries = list_datasets_response["DatasetSummaries"]
        for dataset_summary in dataset_summaries:
            dataset_list.append(dataset_summary['DatasetName'])
    
    return dataset_list
    
def load_dataset(dataset_name, target_dir):
    """
    This function can be used to download example datasets to run Amazon
    Lookout for Equipment on.
    
    Parameters:
        dataset_name (string):
            Can only be 'expander' at this stage
        target_dir (string):
            Location where to download the data: this location must be readable
            and writable
            
    Returns:
        data (dict): dictionnary with data dataframe, labels dataframe,
        training start and end datetime, evaluation start and end datetime,
        and the tags description dataframe
    """
    if dataset_name == 'expander':
        REGION_NAME = boto3.session.Session().region_name
        BUCKET = f'lookoutforequipmentbucket-{REGION_NAME}'
        PREFIX = 'datasets/demo'
        FILES = ['timeseries.zip', 
                 'labels.csv', 
                 'tags_description.csv', 
                 'timeranges.txt']
        TRAIN_DATA = os.path.join(target_dir, 'training-data')

        os.makedirs(target_dir, exist_ok=True)
        os.makedirs(TRAIN_DATA, exist_ok=True)
        os.makedirs(os.path.join(target_dir, 'label-data'), exist_ok=True)
        os.makedirs(os.path.join(target_dir, 'inference-data'), exist_ok=True)

        root_url = f'https://{BUCKET}.s3.{REGION_NAME}.amazonaws.com/{PREFIX}'
        for f in FILES:
            target_file = os.path.join(target_dir, f)
            url_file = root_url + '/' + f
            urllib.request.urlretrieve(url_file, target_file)

        # Load the time series data:
        timeseries_zip_file = os.path.join(target_dir, 'timeseries.zip')
        with zipfile.ZipFile(timeseries_zip_file, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        os.remove(timeseries_zip_file)
        
        all_tags_fname = os.path.join(target_dir, 'expander.parquet')
        table = pq.read_table(all_tags_fname)
        all_tags_df = table.to_pandas()
        del table
        
        # Load the labels data:
        labels_fname = os.path.join(target_dir, 'labels.csv')
        labels_df = pd.read_csv(labels_fname, header=None)
        labels_df[0] = pd.to_datetime(labels_df[0])
        labels_df[1] = pd.to_datetime(labels_df[1])
        labels_df.columns = ['start', 'end']
        
        # Loads the analysis time range:
        timeranges_fname = os.path.join(target_dir, 'timeranges.txt')
        with open(timeranges_fname, 'r') as f:
            timeranges = f.readlines()
            
        training_start   = pd.to_datetime(timeranges[0][:-1])
        training_end     = pd.to_datetime(timeranges[1][:-1])
        evaluation_start = pd.to_datetime(timeranges[2][:-1])
        evaluation_end   = pd.to_datetime(timeranges[3][:-1])
        
        # Loads the tags description:
        tags_description_fname = os.path.join(target_dir, 'tags_description.csv')
        tags_description_df = pd.read_csv(tags_description_fname)
        
        # Create the training data, by processing each subsystem one by one:
        components = list(tags_description_df['Subsystem'].unique())
        progress_bar = tqdm(components)
        for component in progress_bar:
            progress_bar.set_description(f'Component {component}')
            progress_bar.refresh()
            
            # Check if CSV file already exist and do not overwrite it:
            component_tags_fname = os.path.join(TRAIN_DATA, 
                                                f'{component}', 
                                                f'{component}.csv')
            if not os.path.exists(component_tags_fname):
                # Build the dataframe with all the signal 
                # timeseries for the current subsystem:
                component_tags_list = list(tags_description_df[tags_description_df['Subsystem'] == component]['Tag'])
                component_tags_df = all_tags_df[component_tags_list]
                component_tags_df = component_tags_df.reset_index()
                component_tags_df['Timestamp'] = component_tags_df['Timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S.%f')
                
                # Save to disk:
                os.makedirs(os.path.join(TRAIN_DATA, f'{component}'), exist_ok=True)
                component_tags_df.to_csv(component_tags_fname, index=None)
        
        # Build a dictionnary with all the data:
        return {
            'data': all_tags_df,
            'labels': labels_df,
            'tags_description': tags_description_df,
            'training_start': training_start,
            'training_end': training_end,
            'evaluation_start': evaluation_start,
            'evaluation_end': evaluation_end
        }

    else:
        raise Exception('Dataset name must be one of ["expander"]')
        
def upload_dataset(root_dir, bucket, prefix):
    """
    Upload a local dataset to S3. This method will look for a `training-data`
    and a `label-data` in the root_dir passed in argument and upload all
    the content from both these folders to S3.
    
    Parameters:
        root_dir (string):
            Path to the local data
        bucket (string):
            Amazon S3 bucket name
        prefix (string):
            Prefix to a directory on Amazon S3 where to upload the data. This
            prefix *MUST* end with a trailing slash "/"
    """
    for root, _, files in os.walk(os.path.join(root_dir, 'training-data')):
        for f in files:
            component = root.split('/')[-1]
            local_file = os.path.join(root, f)
            upload_file_to_s3(
                local_file, 
                bucket, 
                prefix + f'training-data/{component}/' + f
            )
            
    label_file = os.path.join(root_dir, 'labels.csv')
    upload_file_to_s3(label_file, bucket, prefix + 'label-data/labels.csv')
    
def upload_file_to_s3(file_name, bucket, prefix=None):
    """
    Upload a file to an S3 bucket

    Parameters:
        file_name (string):
            Local path to the file to upload
        bucket (string):
            Bucket to upload to
        prefix (string)
            S3 object name. If not specified then file_name is used
    
    Returns:
        (boolean)
            True if file was uploaded, else False
    """
    # If S3 object_name was not specified, use file_name
    if prefix is None:
        prefix = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(file_name, bucket, prefix)
        
    except ClientError as e:
        print(e)
        return False
        
    return True
    
def prepare_inference_data(root_dir,
                           sample_data_dict,
                           bucket,
                           prefix,
                           num_sequences=3, 
                           frequency=5,
                           start_date=None):
    """
    This function prepares sequence of data suitable as input for an inference
    scheduler.
    
    Parameters:
        root_dir (string):
            Location where the inference data will be written
        sample_data_dict (dict):
            A dictionnary with the sample data as output by `load_dataset()`
            method
        bucket (string):
            Amazon S3 bucket name
        prefix (string):
            Prefix to a directory on Amazon S3 where to upload the data. This
            prefix *MUST* end with a trailing slash "/"
        num_sequences (integer):
            Number of short time series sequences to extract: each sequence
            will be used once by a scheduler. Defaults to 3: a scheduler will
            run 3 times before failing (unless you provide additional suitable
            files in the input location)
        frequency (integer):
            The scheduling frequency in minutes: this **MUST** match the 
            resampling rate used to train the model (defaults to 5 minutes)
        start_date (string or datetime):
            The datetime to start the extraction from. Default is None: in this
            case this method will start looking at date located at the beginning
            of the evaluation period associated to this sample
    """
    tags_df = sample_data_dict['data']
    tags_description_df = sample_data_dict['tags_description']
    components = tags_description_df['Subsystem'].unique()
    os.makedirs(os.path.join(root_dir, 'inference-data', 'input'), exist_ok=True)
    
    # If no start date is provided we take 
    # the first one in the evaluation data:
    if start_date is None:
        start = sample_data_dict['evaluation_start']
    elif isinstance(start_date, str):
        start = pd.to_datetime(start_date)
    else:
        start = start_date
    
    # Loops through each sequence to extract
    for i in range(num_sequences):
        end = start + timedelta(minutes=+frequency - 1)
        
        # Rounding time to the previous 5 minutes:
        tm = datetime.now()
        tm = tm - timedelta(
            minutes=tm.minute % frequency,
            seconds=tm.second,
            microseconds=tm.microsecond
        )
        tm = tm + timedelta(minutes=+frequency * (i))
        current_timestamp = (tm).strftime(format='%Y%m%d%H%M%S')
    
        # For each sequence, we need to loop through all components:
        print(f'Extracting data from {start} to {end}')
        new_index = None
        for component in components:
            # Extracting the dataframe for this component and this particular time range:
            signals = list(tags_description_df.loc[(tags_description_df['Subsystem'] == component), 'Tag'])
            signals_df = tags_df.loc[start:end, signals]
            
            # We need to reset the index to match the time 
            # at which the scheduler will run inference:
            if new_index is None:
                new_index = pd.date_range(
                    start=tm,
                    periods=signals_df.shape[0], 
                    freq=f'{frequency}min'
                )
            signals_df.index = new_index
            signals_df.index.name = 'Timestamp'
            signals_df = signals_df.reset_index()
    
            # Export this file in CSV format:
            component_fname = os.path.join(root_dir, 'inference-data', 'input', f'{component}_{current_timestamp}.csv')
            signals_df.to_csv(component_fname, index=None)
            
            # Upload this file to S3:
            upload_file_to_s3(
                component_fname,
                bucket,
                f'{prefix}inference-data/input/{component}_{current_timestamp}.csv'
            )
        
        start = start + timedelta(minutes=+frequency)
        
def delete_dataset(dataset_name, delete_children=False, verbose=False):
    """
    This method hierarchically delete a dataset and all the associated
    children: models and schedulers.
    
    Parameters:
        dataset_name (string):
            Name of the dataset to delete
        delete_children (boolean):
            If True, will delete all the children resource (Default: False)
        verbose (boolean):
            If True, will print messages about the resource deleted
            (Default: False)
    """
    client = boto3.client('lookoutequipment')
    
    response = client.list_datasets(DatasetNameBeginsWith=dataset_name)
    num_datasets = len(response['DatasetSummaries'])

    if num_datasets > 0:
        response = client.list_models(DatasetNameBeginsWith=dataset_name)
        num_models = len(response['ModelSummaries'])
        print(f'{num_models} model(s) found for this dataset')

        if (num_models > 0) and (delete_children == True):
            for model_summaries in response['ModelSummaries']:
                model_name = model_summaries['ModelName']
                if verbose:
                    print(f'- Model {model_name}: DELETING')

                response = client.list_inference_schedulers(ModelName=model_name)
                num_schedulers = len(response['InferenceSchedulerSummaries'])

                if num_schedulers > 0:
                    # Stopping and deleting all the schedulers:
                    for scheduler_summary in response['InferenceSchedulerSummaries']:
                        scheduler_name = scheduler_summary['InferenceSchedulerName']
                        scheduler_arn = scheduler_summary['InferenceSchedulerArn']
                        
                        if verbose:
                            print(f'- Scheduler {scheduler_name}: DELETING')
                        client.stop_inference_scheduler(InferenceSchedulerName=scheduler_name)
                        status = ''
                        while status != 'STOPPED':
                            response = client.describe_inference_scheduler(InferenceSchedulerName=scheduler_name)
                            status = response['Status']
                            time.sleep(10)

                        client.delete_inference_scheduler(InferenceSchedulerName=scheduler_name)

                    # Waiting loop until all the schedulers are gone:
                    while num_schedulers > 0:
                        response = client.list_inference_schedulers(ModelName=model_name)
                        num_schedulers = len(response['InferenceSchedulerSummaries'])
                        time.sleep(10)

                client.delete_model(ModelName=model_name)

        elif (num_models > 0) and (delete_children == False) and (verbose):
            print((
                'Some models have been trained with this dataset. '
                'You need to delete them before you can safely '
                'delete this dataset'
            ))

        # Waiting for all models to be deleted before moving forward with dataset deletion:
        while num_models > 0:
            response = client.list_models(DatasetNameBeginsWith=dataset_name)
            num_models = len(response['ModelSummaries'])
            time.sleep(10)

        if verbose:
            print(f'- Dataset: DELETING')
        client.delete_dataset(DatasetName=dataset_name)
        time.sleep(2)
        
        if verbose:
            print('Done')

    else:
        print(f'No dataset with this name ({dataset_name}) found.')

class LookoutEquipmentDataset:
    """
    A class to manage Lookout for Equipment datasets
    """
    def __init__(self, 
                 dataset_name,
                 access_role_arn,
                 component_fields_map=None,
                 component_root_dir=None):
        """
        Create a new instance to configure all the attributes necessary to 
        manage a Lookout for Equipment dataset.
        
        Parameters:
            dataset_name (string):
                the name of the dataset to manage
            component_fields_map (string):
                the mapping of the different fields associated to this
                dataset. Either ``component_root_dir`` or 
                ``component_fields_map`` must be provided. Defaults to None.
            component_root_dir (string):
                the root location where the sensor data are stored. Either
                ``component_root_dir`` or ``component_fields_map`` must be 
                provided. Defaults to None. Can be a local folder or an S3
                location.
            access_role_arn (string):
                the ARN of a role that will allow Lookout for Equipment to 
                read data from the data source bucket on S3
        """
        # Parameters consistency checks:
        if (component_fields_map is None) and (component_root_dir is None):
            raise Exception((
                '``component_root_dir`` and ``component_fields_map``'
                ' cannot both be set to None'
            ))
        
        # Schema definition:
        if component_fields_map is not None:
            self._dataset_schema = create_data_schema(component_fields_map)
        elif component_root_dir is not None:
            if component_root_dir[:5] == 's3://':
                self._dataset_schema = create_data_schema_from_s3_path(component_root_dir)
                
            else:
                self._dataset_schema = create_data_schema_from_dir(component_root_dir)

        # Initializing other attributes:
        self._dataset_name = dataset_name
        self.client = boto3.client('lookoutequipment')
        self._role_arn = access_role_arn
        self._ingestion_job_id = None
        self._ingestion_response = {}
        self._components_list = None
        self._schema = None
        
    def create(self):
        """
        Creates a Lookout for Equipment dataset

        Returns:
            string:
                Response of the create dataset API
        """
        # Checks if the dataset already exists:
        list_datasets_response = self.client.list_datasets(
            DatasetNameBeginsWith=self._dataset_name
        )

        dataset_exists = False
        for dataset_summary in list_datasets_response['DatasetSummaries']:
            if dataset_summary['DatasetName'] == self._dataset_name:
                dataset_exists = True
                break
    
        # If the dataset exists we just returns that message:
        if dataset_exists:
            print((f'Dataset "{self._dataset_name}" already exists and can be '
                    'used to ingest data or train a model.'))
    
        # Otherwise, we create it:
        else:
            print(f'Dataset "{self._dataset_name}" does not exist, creating it...\n')
    
            try:
                create_dataset_response = self.client.create_dataset(
                    DatasetName=self._dataset_name,
                    DatasetSchema={
                        'InlineDataSchema': self._dataset_schema
                    }
                )
                
                return create_dataset_response

            except Exception as e:
                print(e)
                
    def list_models(self):
        """
        List all the models trained with this dataset
        
        Returns:
            list of strings:
                A list with the names of every models trained with this dataset
        """
        models_list = list_models(
            dataset_name_prefix=self._dataset_name
        )
        
        return models_list
    
    def delete(self, force_delete=True):
        """
        Deletes the dataset
        
        Parameters:
            force_delete (boolean):
                if set to True, also delete all the models that are using this
                dataset before deleting it. Otherwise, this method will list
                the attached models (Default: True)
        """
        # Let's try to delete this dataset:
        try:
            delete_dataset_response = self.client.delete_dataset(
                DatasetName=self._dataset_name
            )
            print(f'Dataset "{self._dataset_name}" is deleted successfully.')
            return delete_dataset_response
            
        # This might not work if the dataset 
        # is used by an existing trained model:
        except Exception as e:
            error_code = e.response['Error']['Code']
            
            # If the dataset is used by existing models and we asked a
            # forced delete, we also delete the associated models before
            # trying again the dataset deletion:
            if (error_code == 'ConflictException') and (force_delete):
                print(('Dataset is used by at least a model, deleting the '
                      'associated model(s) before deleting dataset.'))
                models_list = self.list_models()
    
                # List all the models that use this dataset and delete them:
                for model_to_delete in models_list:
                    self.client.delete_model(ModelName=model_to_delete)
                    print(f'- Model "{model_to_delete}" is deleted successfully.')
                    
                # Retry the dataset deletion
                delete_dataset_response = self.client.delete_dataset(
                    DatasetName=self._dataset_name
                )
                print(f'Dataset "{self._dataset_name}" is deleted successfully.')
                return delete_dataset_response
                
            # If force_delete is set to False, then we only list the models
            # using this dataset back to the user:
            elif force_delete == False:
                print('Dataset is used by the following models:')
                models_list = self.list_models()
    
                for model_name in models_list:
                    print(f'- {model_name}')
                    
                print(('Rerun this method with `force_delete` set '
                       'to True to delete these models'))
    
            # Dataset name not found:
            elif (error_code == 'ResourceNotFoundException'):
                print((f'Dataset "{self._dataset_name}" not found: creating a '
                        'dataset with this name is possible.'))
    
    def ingest_data(self, bucket, prefix, wait=False, sleep_time=60):
        """
        Ingest data from an S3 location into the dataset
        
        Parameters:
            bucket (string):
                Bucket name where the data to ingest are located
            prefix (string):
                Actual location inside the aforementioned bucket
            wait (Boolean):
                If True, this function will wait for the ingestion to finish
                (default to False)
            sleep_time (integer)
                how many seconds should we wait before polling again when the
                ``wait`` parameter is True (default: 60)
                
        Returns:
            string:
                Response of the start ingestion job API call (if ``wait`` is
                False) or of the actual finished ingestion job (if ``wait``
                is True)
        """
        # Configure the input location:
        ingestion_input_config = dict()
        ingestion_input_config['S3InputConfiguration'] = dict([
            ('Bucket', bucket),
            ('Prefix', prefix)
        ])

        # Start data ingestion:
        start_data_ingestion_job_response = self.client.start_data_ingestion_job(
            DatasetName=self._dataset_name,
            RoleArn=self._role_arn, 
            IngestionInputConfiguration=ingestion_input_config
        )
        self._ingestion_job_id = start_data_ingestion_job_response['JobId']
        
        # If the user wants to wait for the ingestion to finish:
        if wait == True:
            self.poll_data_ingestion(sleep_time=sleep_time)
            data_ingestion_job_response = self.client.describe_data_ingestion_job(
                JobId=self._ingestion_job_id
            )
            return data_ingestion_job_response
            
        # Otherwise, we return interactively and let the ingestion job 
        # continue in the background:
        else:
            return start_data_ingestion_job_response
    
    def poll_data_ingestion(self, sleep_time=60):
        """
        This function polls the data ingestion describe API and prints a status 
        until the ingestion is done.
        
        Parameters:
            sleep_time (integer)
                How many seconds should we wait before polling again
                (default: 60)
        """
        ingestion_response = self.client.describe_data_ingestion_job(
            JobId=self._ingestion_job_id
        )
        
        status = ingestion_response['Status']
        while status == 'IN_PROGRESS':
            time.sleep(sleep_time)
            ingestion_response = self.client.describe_data_ingestion_job(
                JobId=self._ingestion_job_id
            )
            status = ingestion_response['Status']
            print(
                str(pd.to_datetime(datetime.now()))[:19],
                "| Data ingestion:", 
                status
            )
            
    def get_component_field_map(self, component):
        """
        """
        if component not in self.components_list:
            raise Exception(f'Component "{component}" not part of dataset schema. Valid components are {self.components_list}')
            
        for field_map in self.schema['Components']:
            if field_map['ComponentName'] == component:
                break
                
        field_map = {
            field_map['ComponentName']: 
            [tag['Name'] for tag in field_map['Columns']]
        }
        
        return field_map
             
    # -------
    # Getters
    # -------
    @property
    def dataset_name(self):
        """
        ``string`` with the name given to the dataset
        """
        return self.dataset_name
        
    @property
    def dataset_schema(self):
        """
        ``string`` with a JSON-formatted string describing the data schema the 
        dataset must conform to
        """
        return self._dataset_schema
        
    @property
    def role_arn(self):
        """
        ``string`` containing the role ARN necessary to access the S3 location
        where the datasets are stored
        """
        return self._role_arn
    
    @property
    def ingestion_job_id(self):
        """
        ``string`` the ID of the data ingestion job
        """
        if self._ingestion_job_id is not None:
            self._ingestion_response = self.client.describe_data_ingestion_job(
                JobId=self._ingestion_job_id
            )
        
        return self._ingestion_job_id
    
    @property
    def components_list(self):
        """
        ``list`` of components part of the schema of this dataset
        """
        response = self.client.describe_dataset(
            DatasetName=self._dataset_name
        )
        
        c_list = []
        for c in eval(response['Schema'])['Components']:
            c_list.append(c['ComponentName'])
            
        self._components_list = c_list
        
        return self._components_list
    
    @property
    def schema(self):
        """
        ``dict`` dictionnary containing the schema of this dataset if it was
        already created in Lookout for Equipment
        """
        response = self.client.describe_dataset(
            DatasetName=self._dataset_name
        )
        self._schema = eval(response['Schema'])
        
        return self._schema
    
    @property
    def ingestion_job_response(self):
        """
        ``string`` with a JSON-formatted string describing the response details
        of a data ingestion job. Useful to inspect error message when ingestion
        fails
        """
        if (len(self._ingestion_response) == 0 and 
                self._ingestion_job_id is not None):
            self._ingestion_response = self.client.describe_data_ingestion_job(
                JobId=self._ingestion_job_id
            )
            
        return self._ingestion_response