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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib import gridspec
from scipy.stats import wasserstein_distance
from tqdm import tqdm

class LookoutEquipmentAnalysis:
    """
    A class to manage Lookout for Equipment result analysis
    
    Attributes:
        model_name (string): the name of the Lookout for Equipment trained model
        predicted_ranges (pandas.DataFrame): a Pandas dataframe with the 
            predicted anomaly ranges listed in chronological order with a Start 
            and End columns
        labelled_ranges (pandas.DataFrame): A Pandas dataframe with the labelled
            anomaly ranges listed in chronological order with a Start and End 
            columns
        df_list (list of pandas.DataFrame): A list with each time series into a 
            dataframe
    """
    def __init__(self, model_name, tags_df):
        """
        Create a new analysis for a Lookout for Equipment model.
        
        Parameters:
            model_name (string):
                The name of the Lookout for Equipment trained model
                
            tags_df (pandas.DataFrame):
                A dataframe containing all the signals, indexed by time
                
            region_name (string):
                Name of the AWS region from where the service is called.
        """
        self.client = boto3.client('lookoutequipment')
        self.model_name = model_name
        self.predicted_ranges = None
        self.labelled_ranges = None
        
        self.ts_normal_training = None
        self.ts_label_evaluation = None
        self.ts_known_anomalies = None
        
        self.df_list = dict()
        for signal in tags_df.columns:
            self.df_list.update({signal: tags_df[[signal]]})
            
        model_description = self.client.describe_model(ModelName=self.model_name)
        if model_description['Status'] == 'FAILED':
            raise Exception('Model training failed, nothing to analyze.')
        
        # Extracting time ranges used at training time:
        self.training_start = pd.to_datetime(
            model_description['TrainingDataStartTime'].replace(tzinfo=None)
        )
        self.training_end = pd.to_datetime(
            model_description['TrainingDataEndTime'].replace(tzinfo=None)
        )
        self.evaluation_start = pd.to_datetime(
            model_description['EvaluationDataStartTime'].replace(tzinfo=None)
        )
        self.evaluation_end = pd.to_datetime(
            model_description['EvaluationDataEndTime'].replace(tzinfo=None)
        )

    def _load_model_response(self):
        """
        Use the trained model description to extract labelled and predicted 
        ranges of anomalies. This method will extract them from the 
        DescribeModel API from Lookout for Equipment and store them in the
        labelled_ranges and predicted_ranges properties.
        """
        describe_model_response = self.client.describe_model(
            ModelName=self.model_name
        )
        
        if self.labelled_ranges is None:
            self.labelled_ranges = eval(
                describe_model_response['ModelMetrics']
            )['labeled_ranges']
            if len(self.labelled_ranges) > 0:
                self.labelled_ranges = pd.DataFrame(self.labelled_ranges)
                self.labelled_ranges['start'] = pd.to_datetime(self.labelled_ranges['start'])
                self.labelled_ranges['end'] = pd.to_datetime(self.labelled_ranges['end'])
                
            else:
                self.labelled_ranges = pd.DataFrame(columns=['start', 'end'])
            
        self.predicted_ranges = eval(
            describe_model_response['ModelMetrics']
        )['predicted_ranges']
        if len(self.predicted_ranges) > 0:
            self.predicted_ranges = pd.DataFrame(self.predicted_ranges)
            self.predicted_ranges['start'] = pd.to_datetime(self.predicted_ranges['start'])
            self.predicted_ranges['end'] = pd.to_datetime(self.predicted_ranges['end'])
            
        else:
            self.predicted_ranges = pd.DataFrame(columns=['start', 'end'])
        
    def set_time_periods(
        self, 
        evaluation_start, 
        evaluation_end, 
        training_start, 
        training_end
    ):
        """
        Set the time period of analysis
        
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
        self.evaluation_start = evaluation_start
        self.evaluation_end = evaluation_end
        self.training_start = training_start
        self.training_end = training_end
    
    def get_predictions(self):
        """
        Get the anomaly ranges predicted by the current model
        
        Returns:
            pandas.DataFrame:
                A Pandas dataframe with the predicted anomaly ranges listed in
                chronological order with a Start and End columns
        """
        if self.predicted_ranges is None:
            self._load_model_response()
            
        return self.predicted_ranges
        
    def get_labels(self, labels_fname=None):
        """
        Get the labelled ranges as provided to the model before training
        
        Parameters:
            labels_fname (string):
                As an option, if you provide a path to a CSV file containing
                the label ranges, this method will use this file to load the
                labels. If this argument is not provided, it will load the
                labels from the trained model Describe API  (Default to None)
        
        Returns:
            pandas.DataFrame:
                A Pandas dataframe with the labelled anomaly ranges listed in
                chronological order with a Start and End columns
        """
        if labels_fname is not None:
            labels_df = pd.read_csv(labels_fname, header=None)
            labels_df[0] = pd.to_datetime(labels_df[0])
            labels_df[1] = pd.to_datetime(labels_df[1])
            labels_df.columns = ['start', 'end']
            self.labelled_ranges = labels_df
        
        elif self.labelled_ranges is None:
            self._load_model_response()
            
        return self.labelled_ranges
    
    def _get_time_ranges(self):
        """
        Extract DateTimeIndex with normal values and anomalies from the
        predictions generated by the model.
        
        Returns:
            pandas.DateTimeIndex:
                Timestamp index for all the values marked as normal during the
                training period
            pandas.DateTimeIndex:
                Timestamp index for all the values predicted as anomalies by
                the model during the evaluation period
        """
        # Extract the first time series 
        tag = list(self.df_list.keys())[0]
        tag_df = self.df_list[tag]
        
        # Initialize the predictions dataframe:
        predictions_df = pd.DataFrame(columns=['Prediction'], index=tag_df.index)
        predictions_df['Prediction'] = 0

        # Loops through the predicted and labelled anomalies
        # ranges and set these predictions to 1 (predicted) 
        # or 2 (initially known):
        for index, row in self.predicted_ranges.iterrows():
            predictions_df.loc[row['start']:row['end'], 'Prediction'] = 1
        for index, row in self.labelled_ranges.iterrows():
            predictions_df.loc[row['start']:row['end'], 'Prediction'] = 2

        # Limits the analysis range to the evaluation period:
        predictions_df = predictions_df[self.training_start:self.evaluation_end]
        
        # Build a DateTimeIndex for normal values and anomalies:
        index_normal = predictions_df[predictions_df['Prediction'] == 0].index
        index_anomaly = predictions_df[predictions_df['Prediction'] == 1].index
        index_known = predictions_df[predictions_df['Prediction'] == 2].index
        
        return index_normal, index_anomaly, index_known
    
    def compute_histograms(
        self, 
        index_normal=None, 
        index_anomaly=None, 
        num_bins=20
    ):
        """
        This method loops through each signal and computes two distributions of
        the values in the time series: one for all the anomalies found in the
        evaluation period and another one with all the normal values found in the
        same period. It then computes the Wasserstein distance between these two
        histograms and then rank every signals based on this distance. The higher
        the distance, the more different a signal is when comparing anomalous
        and normal periods. This can orient the investigation of a subject 
        matter expert towards the sensors and associated components.
        
        Parameters:
            index_normal (pandas.DateTimeIndex):
                All the normal indices
                
            index_anomaly (pandas.DateTimeIndex):
                All the indices for anomalies
                
            num_bins (integer):
                Number of bins to use to build the distributions (default: 20)
        """
        if (index_normal is None) or (index_anomaly is None):
            index_lists = self._get_time_ranges()
            self.ts_normal_training = index_lists[0]
            self.ts_label_evaluation = index_lists[1]
            self.ts_known_anomalies = index_lists[2]

        self.num_bins = num_bins

        # Now we loop on each signal to compute a 
        # histogram of each of them in this anomaly range,
        # compte another one in the normal range and
        # compute a distance between these:
        rank = dict()
        for tag, current_tag_df in tqdm(
            self.df_list.items(), 
            desc='Computing distributions'
        ):
            try:
                # Get the values for the whole signal, parts
                # marked as anomalies and normal part:
                current_signal_values = current_tag_df[tag]
                current_signal_evaluation = current_tag_df.loc[self.ts_label_evaluation, tag]
                current_signal_training = current_tag_df.loc[self.ts_normal_training, tag]

                # Let's compute a bin width based on the whole range of possible 
                # values for this signal (across the normal and anomalous periods).
                # For both normalization and aesthetic reasons, we want the same
                # number of bins across all signals:
                bin_width = (np.max(current_signal_values) - np.min(current_signal_values))/self.num_bins
                bins = np.arange(
                    np.min(current_signal_values), 
                    np.max(current_signal_values) + bin_width, 
                    bin_width
                )

                # We now use the same bins arrangement for both parts of the signal:
                u = np.histogram(
                    current_signal_training, 
                    bins=bins, 
                    density=True
                )[0]
                v = np.histogram(
                    current_signal_evaluation, 
                    bins=bins, 
                    density=True
                )[0]

                # Wasserstein distance is the earth mover distance: it can be 
                # used to compute a similarity between two distributions: this
                # metric is only valid when the histograms are normalized (hence
                # the density=True in the computation above):
                d = wasserstein_distance(u, v)
                rank.update({tag: d})

            except Exception as e:
                rank.update({tag: 0.0})

        # Sort histograms by decreasing Wasserstein distance:
        rank = {k: v for k, v in sorted(rank.items(), key=lambda rank: rank[1], reverse=True)}
        self.rank = rank
        
    def plot_histograms_v2(self, custom_ranking, nb_cols=3, max_plots=12, num_bins=20):
        index_lists = self._get_time_ranges()
        self.ts_normal_training = index_lists[0]
        self.ts_label_evaluation = index_lists[1]
        self.ts_known_anomalies = index_lists[2]
        self.num_bins = num_bins
            
        # Prepare the figure:
        nb_rows = len(self.df_list.keys()) // nb_cols + 1
        plt.style.use('Solarize_Light2')
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        fig = plt.figure(figsize=(16, int(nb_rows * 3)))
        gs = gridspec.GridSpec(nb_rows, nb_cols, hspace=0.5, wspace=0.25)
        axes = []
        for i in range(max_plots):
            axes.append(fig.add_subplot(gs[i]))

        # Loops through each signal by decreasing distance order:
        i = 0
        for tag, current_rank in tqdm(
            custom_ranking.items(), 
            total=max_plots, 
            desc='Preparing histograms'
        ):
            # We stop after reaching the number of plots we are interested in:
            if i > max_plots - 1:
                break

            try:
                # Get the anomaly and the normal values from the current signal:
                current_signal_values = self.df_list[tag][tag]
                
                current_signal_evaluation = self.df_list[tag].loc[self.ts_label_evaluation, tag]
                current_signal_training = self.df_list[tag].loc[self.ts_normal_training, tag]
                
                

                # Compute the bin width and bin edges to match the 
                # number of bins we want to have on each histogram:
                bin_width =(np.max(current_signal_values) - np.min(current_signal_values))/self.num_bins
                bins = np.arange(
                    np.min(current_signal_values), 
                    np.max(current_signal_values) + bin_width, 
                    bin_width
                )

                # Add both histograms in the same plot:
                axes[i].hist(current_signal_training, 
                         density=True, 
                         alpha=0.5, 
                         color=colors[1], 
                         bins=bins, 
                         edgecolor='#FFFFFF')
                axes[i].hist(current_signal_evaluation, 
                         alpha=0.5, 
                         density=True, 
                         color=colors[5], 
                         bins=bins, 
                         edgecolor='#FFFFFF')

            except Exception as e:
                print(e)
                axes[i] = plt.subplot(gs[i])

            # Removes all the decoration to leave only the histograms:
            axes[i].grid(False)
            axes[i].get_yaxis().set_visible(False)
            axes[i].get_xaxis().set_visible(False)

            # Title will be the tag name followed by the score:
            title = tag
            title += f' (score: {current_rank:.02f}%)'
            axes[i].set_title(title, fontsize=10)

            i+= 1
            
        return fig, axes
        
    def plot_histograms(self, nb_cols=3, max_plots=12):
        """
        Once the histograms are computed, we can plot the top N by decreasing 
        ranking distance. By default, this will plot the histograms for the top
        12 signals, with 3 plots per line.
        
        Parameters:
            nb_cols (integer):
                Number of plots to assemble on a given row (default: 3)
            max_plots (integer):
                Number of signal to consider (default: 12)
                
        Returns:
            tuple: tuple containing:
                * A ``matplotlib.pyplot.figure`` where the plots are drawn
                * A ``list of matplotlib.pyplot.Axis`` with each plot drawn here
        """
        # Prepare the figure:
        nb_rows = len(self.df_list.keys()) // nb_cols + 1
        plt.style.use('Solarize_Light2')
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        fig = plt.figure(figsize=(16, int(nb_rows * 3)))
        gs = gridspec.GridSpec(nb_rows, nb_cols, hspace=0.5, wspace=0.25)
        axes = []
        for i in range(max_plots):
            axes.append(fig.add_subplot(gs[i]))

        # Loops through each signal by decreasing distance order:
        i = 0
        for tag, current_rank in tqdm(
            self.rank.items(), 
            total=max_plots, 
            desc='Preparing histograms'
        ):
            # We stop after reaching the number of plots we are interested in:
            if i > max_plots - 1:
                break

            try:
                # Get the anomaly and the normal values from the current signal:
                current_signal_values = self.df_list[tag][tag]
                current_signal_evaluation = self.df_list[tag].loc[self.ts_label_evaluation, tag]
                current_signal_training = self.df_list[tag].loc[self.ts_normal_training, tag]

                # Compute the bin width and bin edges to match the 
                # number of bins we want to have on each histogram:
                bin_width =(np.max(current_signal_values) - np.min(current_signal_values))/self.num_bins
                bins = np.arange(
                    np.min(current_signal_values), 
                    np.max(current_signal_values) + bin_width, 
                    bin_width
                )

                # Add both histograms in the same plot:
                axes[i].hist(current_signal_training, 
                         density=True, 
                         alpha=0.5, 
                         color=colors[1], 
                         bins=bins, 
                         edgecolor='#FFFFFF')
                axes[i].hist(current_signal_evaluation, 
                         alpha=0.5, 
                         density=True, 
                         color=colors[5], 
                         bins=bins, 
                         edgecolor='#FFFFFF')

            except Exception as e:
                print(e)
                axes[i] = plt.subplot(gs[i])

            # Removes all the decoration to leave only the histograms:
            axes[i].grid(False)
            axes[i].get_yaxis().set_visible(False)
            axes[i].get_xaxis().set_visible(False)

            # Title will be the tag name followed by the score:
            title = tag
            title += f' (score: {current_rank:.02f})'
            axes[i].set_title(title, fontsize=10)

            i+= 1
            
        return fig, axes
            
    def plot_signals(self, nb_cols=3, max_plots=12):
        """
        Once the histograms are computed, we can plot the top N signals by 
        decreasing ranking distance. By default, this will plot the signals for 
        the top 12 signals, with 3 plots per line. For each signal, this method
        will plot the normal values in green and the anomalies in red.
        
        Parameters:
            nb_cols (integer):
                Number of plots to assemble on a given row (default: 3)
            max_plots (integer):
                Number of signal to consider (default: 12)
                
        Returns:
            tuple: tuple containing:
                * A ``matplotlib.pyplot.figure`` where the plots are drawn
                * A ``list of matplotlib.pyplot.Axis`` with each plot drawn here
        """
        # Prepare the figure:
        nb_rows = max_plots // nb_cols + 1
        plt.style.use('Solarize_Light2')
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        fig = plt.figure(figsize=(28, int(nb_rows * 4)))
        gs = gridspec.GridSpec(nb_rows, nb_cols, hspace=0.5, wspace=0.25)
        axes = []
        for i in range(max_plots):
            axes.append(fig.add_subplot(gs[i]))
        
        # Loops through each signal by decreasing distance order:
        i = 0
        for tag, current_rank in self.rank.items():
            # We stop after reaching the number of plots we are interested in:
            if i > max_plots - 1:
                break

            # Get the anomaly and the normal values from the current signal:
            current_signal_evaluation = self.df_list[tag].loc[self.ts_label_evaluation, tag]
            current_signal_training = self.df_list[tag].loc[self.ts_normal_training, tag]
            current_signal_known = self.df_list[tag].loc[self.ts_known_anomalies, tag]

            # Plot both time series with a line plot
            # axes.append(plt.subplot(gs[i]))
            axes[i].plot(current_signal_training, 
                         linewidth=0.5, 
                         alpha=0.8, 
                         color=colors[1])
            axes[i].plot(current_signal_evaluation, 
                         linewidth=0.5, 
                         alpha=0.8, 
                         color=colors[5])
            axes[i].plot(current_signal_known, 
                         linewidth=0.5, 
                         alpha=0.8, 
                         color='#AAAAAA')

            # Title will be the tag name followed by the score:
            title = tag
            title += f' (score: {current_rank:.02f})'
                
            axes[i].set_title(title, fontsize=10)
            start = min(
                min(self.ts_label_evaluation),
                min(self.ts_normal_training), 
                min(self.ts_known_anomalies)
            )
            end = max(
                max(self.ts_label_evaluation),
                max(self.ts_normal_training), 
                max(self.ts_known_anomalies)
            )
            axes[i].set_xlim(start, end)

            i += 1
            
        return fig, axes
            
    def get_ranked_list(self, max_signals=12):
        """
        Returns the list of signals with computed rank.
        
        Parameters:
            max_signals (integer)
                Number of signals to consider (default: 12)
        
        Returns:
            pandas.DataFrame:
                A dataframe with each signal and the associated rank value
        """
        significant_signals_df = pd.DataFrame(list(self.rank.items())[:max_signals])
        significant_signals_df.columns = ['Tag', 'Rank']
        
        return significant_signals_df