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
import datetime
import markdown
import pandas as pd
import pytz
import s3fs
import time

class LookoutEquipmentSchedulerInspector:
    """
    A class to be used to inspect existing inference scheduler and output a
    report about how the inputs should be structured
    
    Attributes:
        scheduler_name (string):
            Name of the scheduler to inspect
        delay_offset (integer):
            Data delay offset of the scheduler
        frequency (integer):
            Data upload frequency of the scheduler
        timestamp_format (string):
            Timestamp format expected by the scheduler
        delimiter (string):
            Delimiter character between component and timestamp in the 
            name of the input CSV file
        timezone (string):
            In which timezone are you located?
        input_timezone_offset (string):
            Timezone offset between current location and data
        s3_input_location (string):
            Location of the input CSV file to run inference on
        current_time (datetime):
            Current time at which the inspection report is generated
        start_time (datetime):
            Start time the scheduler looks for when looking for valid data
            points to run inference on
        end_time (datetime):
            End time the scheduler looks for when looking for valid data points
            to run inference on
        next_wakeup_time (datetime):
            Computed next time the scheduler will wake up
        next_timestamp (datetime):
            Next timestamp the scheduler will look for in the input file names
    """
    def __init__(self, scheduler_name, current_timezone='UTC'):
        """
        A class to inspect an existing inference scheduler and output a report
        in either Markdown (to be printed in a Jupyter notebook for instance)
        or in an independant HTML file
        
        Parameters:
            scheduler_name (string):
                Name of the scheduler to inspect. This inference scheduler
                must already exist
            current_timezone (string):
                Timezone where the data are generated. Must be one of the
                timezone referenced in the ``pytz.all_timezones`` list
                
        Raises:
            Exception: if a scheduler with this name is not found
        """
        # Initialize a Lookout for Equipment client:
        self.client = boto3.client('lookoutequipment')
        
        # Let's try and find this scheduler: if we find it, we collect the
        # associated model, and dataset and extract the schema the data should
        # conform to:
        try:
            self.description = self.client.describe_inference_scheduler(
                InferenceSchedulerName=scheduler_name
            )
            model_name = self.description['ModelName']
            model_response = self.client.describe_model(ModelName=model_name)
            dataset_name = model_response['DatasetName']
            dataset_response = self.client.describe_dataset(DatasetName=dataset_name)
            self.schema = eval(dataset_response['Schema'])
            self.num_components = len(self.schema)
            
        except Exception as e:
            raise(e)
        
        # Get some scheduler execution parameters:
        self.scheduler_name   = scheduler_name
        self.delay_offset     = self.description['DataDelayOffsetInMinutes']
        self.frequency        = int(self.description['DataUploadFrequency'][2:][:-1])
        self.timestamp_format = self.description['DataInputConfiguration']['InferenceInputNameConfiguration']['TimestampFormat']
        self.delimiter        = self.description['DataInputConfiguration']['InferenceInputNameConfiguration']['ComponentTimestampDelimiter']
        self.timezone         = current_timezone
        
        # Initialize input location and timezone offset
        self._get_input_location()
        self._get_timezone_offset()

    def _get_timezone_offset(self):
        """
        Compute the appropriate timezone offset between the system where the 
        scheduler is configured and the desired timezone where the data are 
        located
        """
        current_timezone = pytz.timezone(self.timezone)
        tz_offset = datetime.datetime.now(current_timezone).strftime('%z')
        self.input_timezone_offset = tz_offset[:3] + ':' + tz_offset[3:]
        
        return self.input_timezone_offset

    def _get_input_location(self):
        """
        This method finds out where the current scheduler should be looking 
        for input files in Amazon S3.
        
        Returns:
            string
                The S3 path where the input files must be positionned so the
                inference scheduler can find them
        """
        s3_input_config = self.description['DataInputConfiguration']['S3InputConfiguration']
        s3_input_location = 's3://' + \
                            s3_input_config['Bucket'] + \
                            s3_input_config['Prefix']
        
        self.s3_input_location = s3_input_location
    
        return self.s3_input_location
        
    def get_next_time_range(self):
        """
        Get the current time and derives the next time the scheduler will wake
        up, what timestamp it will look for to find the right input file to 
        process and which time range to filter out when opening this file.
        """
        # Derive the appropriate time strftime format:
        if self.timestamp_format == 'EPOCH':
            strftime_format = '%s'
        elif self.timestamp_format == 'yyyy-MM-dd-HH-mm-ss':
            strftime_format = '%Y-%m-%d-%H-%M-%S'
        elif self.timestamp_format == 'yyyyMMddHHmmss':
            strftime_format = '%Y%m%d%H%M%S'
            
        current_timezone = pytz.timezone(self.timezone)
        current_time = datetime.datetime.now(current_timezone)
        next_time = current_time - datetime.timedelta(
            minutes=current_time.minute % int(self.frequency),
            seconds=current_time.second,
            microseconds=current_time.microsecond
        )
        next_wakeup_time = next_time + datetime.timedelta(minutes=+self.frequency)
        next_timestamp = (next_time).strftime(format=strftime_format)
        
        start_time = next_time
        end_time = start_time + datetime.timedelta(minutes=+self.frequency, seconds=-1)
                                                  
        self.current_time     = current_time
        self.start_time       = start_time
        self.end_time         = end_time
        self.next_wakeup_time = next_wakeup_time
        self.next_timestamp   = next_timestamp
        
    def build_inspection_report(self):
        """
        Build the inspection report in Markdown format suitable for displaying
        from a Jupyter notebook or any output that can take this format input
        
        Returns:
            string:
                The inspection report in Markdown format
        """
        self.get_next_time_range()
        
        scheduler_description = []
        scheduler_description.append(f'**SCHEDULER: {self.scheduler_name}**\n')
        scheduler_description.append(f'*Scheduler inspection report run at: {self.current_time}*\n')
        scheduler_description.append('Here is the behavior you can expect from this scheduler:\n')
        if self.delay_offset > 0:
            scheduler_description.append(f'* It will wake up every **{self.frequency} minutes**')
            scheduler_description.append(f'and wait for up to **{self.delay_offset} minute(s)** for the data to be available.\n')
        else:
            scheduler_description.append(f'* It will wake up every **{self.frequency} minutes**.\n')
        scheduler_description.append(f'* It will look for CSV files in the following location `{self.s3_input_location}*.csv`.\n')
        scheduler_description.append(f'* The current time is **{self.current_time}** and the next time the scheduler will wake up will be **{self.next_wakeup_time}**\n')
        scheduler_description.append(f'* The dataset associated to this scheduler\'s model has **{self.num_components} components** in its schema.\n')
        scheduler_description.append(f'Each time the scheduler wakes up, it expects to find **{self.num_components} CSV files** in the input location, one for each component as defined in the dataset schema.\n')
        scheduler_description.append(f'If the scheduler was to wake up at **{self.next_wakeup_time}**, it would look for the following files:')
        
        for component in self.schema['Components']:
            component_name = component['ComponentName']
            scheduler_description.append(f'\n* `{component_name}{self.delimiter}{self.next_timestamp}.csv` and this file content would have to follow this template:')
        
            current_table = '\n'
            for col in component['Columns']:
                current_table += ' | ' + col['Name']
            current_table += '|'
            current_table += '\n' + '| --- '*len(component['Columns']) + '|'
            current_table += f'\n| {str(self.start_time)[:19]}' + '| 0.0 '*(len(component['Columns'])-1) + '|'
            current_table += '\n' + '| ... '*len(component['Columns']) + '|'
            current_table += f'\n| {str(self.end_time)[:19]}' + '| 0.0 '*(len(component['Columns'])-1) + '|'
        
            scheduler_description.append(current_table)
            
        self.inspection_report = '\n'.join(scheduler_description)
        return self.inspection_report
        
    def export_to_html(self, html_path=None):
        """
        This method will export the Markdown report as built by the
        ``build_inspection_report()`` method into an HTML file.
        
        Parameters:
            html_path (string):
                path + filename location to write the HTML report to. By 
                default, will write the file to the current location, using the 
                scheduler name and suffixing it with '-inspection-report' 
                (default: None)
        """
        if html_path is None:
            html_path = f'{self.scheduler_name}-inspection-report.html'

        html_header = """<!DOCTYPE html>
<html>
    <head>
        <style>
            body { 
                font-family: "Amazon Ember", Helvetica, Arial, sans-serif; 
                font-size: 14px; 
                background-color: #F2F3F3
            }
            table { 
                border-collapse: collapse;
                padding: 2px 2px 2px 2px;
                margin: 0px 0px 0px 40px;
                font-size: 12px;
                border: 1px solid #EAEDED;
                box-shadow: 0 0 7px #D4D9DA;
            }
            thead { 
                background-color: #FAFAFA;
                text-align: center; 
                height: 20px
            }
            th { padding: 5px 15px 5px 15px }
            td { height: 15px; text-align: right; }
            code { color: #3184C2; font-weight: bold }
            ul { list-style: disc outside none; }
        </style>
    </head>
    
    <body>
        """
        
        html_footer = """
    </body>
</html>
        """

        if self.inspection_report is None:
            _ = self.build_inspection_report()
            
        else:
            with open(html_path, 'w') as f:
                f.writelines(html_header)
                html_string = markdown.markdown(self.inspection_report, extensions=['tables'])
                f.writelines(html_string)
                f.writelines(html_footer)


class LookoutEquipmentScheduler:
    """
    A class to represent a Lookout for Equipment inference scheduler object.
    
    Attributes:
        scheduler_name (string):
            Name of the scheduler associated to this object
        model_name (string):
            Name of the model used to run the inference when the scheduler
            wakes up
        create_request (dict):
            A dictionary containing all the parameters to configure and create
            an inference scheduler
        execution_summaries (list of dict):
            A list of all inference execution results. Each execution is stored
            as a dictionary.
    """
    def __init__(self, scheduler_name, model_name):
        """
        Constructs all the necessary attributes for a scheduler object.
        
        Parameters:
            scheduler_name (string):
                The name of the scheduler to be created or managed
                
            model_name (string):
                The name of the model to schedule inference for
                
            region_name (string):
                Name of the AWS region from where the service is called.
        """
        self.scheduler_name = scheduler_name
        self.model_name = model_name
        self.client = boto3.client('lookoutequipment')
        
        self.create_request = dict()
        self.create_request.update({'ModelName': model_name})
        self.create_request.update({'InferenceSchedulerName': scheduler_name})

        self.execution_summaries = None
        
    def set_parameters(self,
                       input_bucket,
                       input_prefix,
                       output_bucket,
                       output_prefix,
                       role_arn,
                       upload_frequency='PT5M',
                       delay_offset=None,
                       timezone_offset='+00:00',
                       component_delimiter='_',
                       timestamp_format='yyyyMMddHHmmss'
                      ):
        """
        Set all the attributes for the scheduler object.
        
        Parameters:
            input_bucket (string):
                Bucket when the input data are located

            input_prefix (string):
                Location prefix for the input data
                
            output_bucket (string):
                Bucket location for the inference execution output
                
            output_prefix (string):
                Location prefix for the inference result file
                
            role_arn (string):
                Role allowing Lookout for Equipment to read and write data
                from the input and output bucket locations
                
            upload_frequency (string):
                Upload frequency of the data (default: PT5M)
            
            delay_offset (integer):
                Offset in minute, ensuring the data are available when the
                scheduler wakes up to run the inference (default: None)
            
            timezone_offset (string):
                Timezone offset used to match the date in the input filenames
                (default: +00:00)
            
            component_delimiter (string):
                Character to use to delimit component name and timestamp in the
                input filenames (default: '_')
            
            timestamp_format (string):
                Format of the timestamp to look for in the input filenames
                (default: yyyyMMddHHmmss)
        """
        # Configure the mandatory parameters:
        self.create_request.update({'DataUploadFrequency': upload_frequency})
        self.create_request.update({'RoleArn': role_arn})
        
        # Configure the optional parameters:
        if delay_offset is not None:
            self.create_request.update({'DataDelayOffsetInMinutes': delay_offset})
            
        # Setup data input configuration:
        inference_input_config = dict()
        inference_input_config['S3InputConfiguration'] = dict([
            ('Bucket', input_bucket),
            ('Prefix', input_prefix)
        ])
        if timezone_offset is not None:
            inference_input_config['InputTimeZoneOffset'] = timezone_offset
        if component_delimiter is not None or timestamp_format is not None:
            input_name_cfg = dict()
            if component_delimiter is not None:
                input_name_cfg['ComponentTimestampDelimiter'] = component_delimiter
            if timestamp_format is not None:
                input_name_cfg['TimestampFormat'] = timestamp_format
            inference_input_config['InferenceInputNameConfiguration'] = input_name_cfg
        self.create_request.update({
            'DataInputConfiguration': inference_input_config
        })

        #  Set up output configuration:
        inference_output_config = dict()
        inference_output_config['S3OutputConfiguration'] = dict([
            ('Bucket', output_bucket),
            ('Prefix', output_prefix)
        ])
        self.create_request.update({
            'DataOutputConfiguration': inference_output_config
        })
        
    def _poll_event(self, scheduler_status, wait_state, sleep_time=5):
        """
        Wait for a given scheduler update process to be finished
        
        Parameters:
            scheduler_status (string):
                Initial scheduler status
            
            wait_state (string):
                The wait will continue while the status has a value equal
                to this wait_state string (either PENDING, STOPPING)
                
            sleep_time (integer):
                How many seconds should we wait before polling again
                (default: 5)
        """
        print("===== Polling Inference Scheduler Status =====\n")
        print("Scheduler Status: " + scheduler_status)
        while scheduler_status == wait_state:
            time.sleep(sleep_time)
            describe_scheduler_response = self.client.describe_inference_scheduler(
                InferenceSchedulerName=self.scheduler_name
            )
            scheduler_status = describe_scheduler_response['Status']
            print("Scheduler Status: " + scheduler_status)
        print("\n===== End of Polling Inference Scheduler Status =====")

    def create(self, wait=True):
        """
        Create an inference scheduler for a trained Lookout for Equipment model
        
        Parameters:
            wait (boolean):
                Wait for the creation process to finish (default: True)
        """
        # Create the scheduler:
        try:
            create_scheduler_response = self.client.create_inference_scheduler(
                **self.create_request
            )
            
            # Polling scheduler creation status:
            if wait:
                self._poll_event(create_scheduler_response['Status'], 'PENDING')
            
        except Exception as e:
            error_code = e.response['Error']['Code']
            
            # If the scheduler already exists:
            if (error_code == 'ConflictException'):
                print(('This scheduler already exists. Try changing its name'
                       'and retry or try to start it.'))
                
            else:
                raise e

    def start(self, wait=True):
        """
        Start an existing inference scheduler if it exists
        
        Parameters:
            wait (boolean):
                Wait for the starting process to finish (default: True)
        """
        start_scheduler_response = self.client.start_inference_scheduler(
            InferenceSchedulerName=self.scheduler_name
        )

        # Wait until started:
        if wait:
            self._poll_event(start_scheduler_response['Status'], 'PENDING')
        
    def stop(self, wait=True):
        """
        Stop an existing started inference scheduler
        
        Parameters:
            wait (boolean):
                Wait for the stopping process to finish (default: True)
        """
        stop_scheduler_response = self.client.stop_inference_scheduler(
            InferenceSchedulerName=self.scheduler_name
        )

        # Wait until stopped:
        if wait:
            self._poll_event(stop_scheduler_response['Status'], 'STOPPING')
        
    def delete(self):
        """
        Delete the current inference scheduler
        
        Returns:
            dict:
                A JSON dictionary with the response from the delete request API
        """
        if self.get_status() == 'STOPPED':
            delete_scheduler_response = self.client.delete_inference_scheduler(
                InferenceSchedulerName=self.scheduler_name
            )
            
        else:
            raise Exception('Scheduler must be stopped to be deleted.')
        
        return delete_scheduler_response
    
    def get_status(self):
        """
        Get current status of the inference scheduler
        
        Returns:
            string:
                The status of the inference scheduler, as extracted from the
                DescribeInferenceScheduler API
        """
        describe_scheduler_response = self.client.describe_inference_scheduler(
            InferenceSchedulerName=self.scheduler_name
        )
        status = describe_scheduler_response['Status']
        
        return status
    
    def list_inference_executions(self, 
                                  execution_status=None, 
                                  start_time=None, 
                                  end_time=None, 
                                  max_results=50):
        """
        This method lists all the past inference execution triggered by the
        current scheduler.
        
        Parameters:
            execution_status (string):
                Only keep the executions with a given status (default: None)
                
            start_time (pandas.DateTime):
                Filters out the executions that happened before start_time
                (default: None)
                
            end_time (pandas.DateTime):
                Filters out the executions that happened after end_time
                (default: None)
                
            max_results (integer):
                Max number of results you want to get out of this method
                (default: 50)
        
        Returns:
            list of dict:
                A list of all past inference executions, with each inference
                attributes stored in a python dictionary
        """
        # Built the execution request object:
        list_executions_request = {"MaxResults": max_results}
        list_executions_request["InferenceSchedulerName"] = self.scheduler_name
        if execution_status is not None:
            list_executions_request["Status"] = execution_status
        if start_time is not None:
            list_executions_request['DataStartTimeAfter'] = start_time
        if end_time is not None:
            list_executions_request['DataEndTimeBefore'] = end_time

        # Loops through all the inference executed by the current scheduler:
        has_more_records = True
        list_executions = []
        while has_more_records:
            list_executions_response = self.client.list_inference_executions(
                **list_executions_request
            )
            if "NextToken" in list_executions_response:
                list_executions_request["NextToken"] = list_executions_response["NextToken"]
            else:
                has_more_records = False

            list_executions = list_executions + \
                              list_executions_response["InferenceExecutionSummaries"]
                
        # Filter the executions based on their status:
        if execution_status is not None:
            filtered_list_executions = []
            for execution in list_executions:
                if execution['Status'] == execution_status:
                    filtered_list_executions.append(execution)
            
            list_executions = filtered_list_executions

        # Returns all the summaries in a list:
        self.execution_summaries = list_executions
        return list_executions
    
    def get_predictions(self):
        """
        This method loops through all the inference executions and build a
        dataframe with all the predictions generated by the model
        
        Returns:
            pandas.DataFrame:
                A dataframe with one prediction by row (1 for an anomaly or 0
                otherwise). Each row is indexed by timestamp.
        """
        # Fetch the list of execution summaries in case all executions were not captured yet:
        _ = self.list_inference_executions()

        fs = s3fs.S3FileSystem()
        results_json = []
        
        # Loops through the executions summaries:
        for execution_summary in self.execution_summaries:
            # We only get an output if the inference execution is a sucess:
            status = execution_summary['Status']
            if status == 'SUCCESS':
                # Build an S3 path for the JSON-line file:
                bucket = execution_summary['CustomerResultObject']['Bucket']
                key = execution_summary['CustomerResultObject']['Key']
                s3_fname = f's3://{bucket}/{key}'
                
                # Opens the file and concatenate the results into a dataframe:
                with fs.open(s3_fname, 'r') as f:
                    content = [eval(line) for line in f.readlines()]
                    results_json = results_json + content
            
        # Build the final dataframes with all the results:
        if len(results_json) > 0:
            results_df = pd.DataFrame(results_json)
            results_df['timestamp'] = pd.to_datetime(results_df['timestamp'])
            results_df = results_df.set_index('timestamp')
            results_df = results_df.sort_index()
            
            expanded_results = []
            for index, row in results_df.iterrows():
                new_row = dict()
                new_row.update({'timestamp': index})
                new_row.update({'prediction': row['prediction']})
                
                # Models trained before March 28, 2022 do not expose the raw
                # anomaly score so let's check if we actually have it:
                if 'anomaly_score' in results_df.columns:
                    new_row.update({'anomaly_score': row['anomaly_score']})
                
                if row['prediction'] == 1:
                    diagnostics = pd.DataFrame(row['diagnostics'])
                    diagnostics = dict(zip(diagnostics['name'], diagnostics['value']))
                    new_row = {**new_row, **diagnostics}
                    
                expanded_results.append(new_row)
                
            expanded_results = pd.DataFrame(expanded_results)
            expanded_results['timestamp'] = pd.to_datetime(expanded_results['timestamp'])
            expanded_results = expanded_results.set_index('timestamp')
            expanded_results.head()
            
            return expanded_results
            
        else:
            raise Exception('No successful execution found for this scheduler')