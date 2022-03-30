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

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

from matplotlib import gridspec

class TimeSeriesVisualization:
    """
    A class to manage time series visualization along with labels and detected
    events
    """
    DEFAULT_COLORS = {
        'labels': 'tab:green',
        'predictions': 'tab:red'
    }
    
    def __init__(self, 
                 timeseries_df, 
                 data_format, 
                 timestamp_col=None, 
                 tag_col=None,
                 resample=None,
                 verbose=False,
                 ):
        """
        Create a new instance to plot time series with different data structure
        
        Parameters:
            timeseries_df (pandas.DataFrame):
                A dataframe containing time series data that you want to plot
            data_format (string):
                Use "timeseries" if your dataframe has three columns:
                ``timestamp``, ``values`` and ``tagname``. Use "tabular" if
                ``timestamp`` is your first column and all the other tags are
                in the following columns: ``timestamp``, ``tag1``, ``tag2``...
            timestamp_col (string):
                Specifies the name of the columns that contains the 
                timestamps. If set to None, it means the timestamp is already
                an index (default to None)
            tag_col (string):
                If data_format is "timeseries", this argument specifies what
                is the name of the columns that contains the name of the tags
            resample (string):
                If specified, this class will resample the data before plotting
                them. Use the same format than the string rule as used in the
                ``pandas.DataFrame.resample()`` method (default to None)
            verbose (boolean):
                If True, this class will print some messages along the way
                (defaults to False)
        """
        self._data = timeseries_df
        self._format = data_format
        self._timestamp_col = timestamp_col
        self._tag_col = tag_col
        self._tags_list = None
        self._signals_to_plot = []
        self._signals_data = []
        self._tag_split = None
        self._split_labels = []
        self._start_date = None
        self._end_date = None
        self._labels_df = None
        self._predictions_ranges = None
        self._predictions_df = []
        self._predictions_title = []
        self._rolling_average = False
        self._rolling_average_window = None
        self._legend_format = {
            'loc': 'upper right',
            'framealpha': 0.5
        }

        self.verbose = verbose
        self.resample = resample

        # Prepare the figures:
        self.fig_height = 4
        self.height_ratios = []
        self.nb_plots = 0
        self.expanded_results = None
        
        if self._format not in ['timeseries', 'tabular']:
            raise Exception('`data_format` can only either be timeseries or tabular')
        
        if (self._format == 'timeseries') and (self._tag_col is None):
            raise Exception('`tag_col` must be defined when data format is timeseries')
        
    def _build_tags_list(self):
        """
        This method builds the tags list by scanning the dataframe attached
        to this class instance and updates the ``tags_list`` attribute
        """
        if self.format == 'timeseries':
            self._tags_list = self.data[self.tag_col].unique()
            
        elif self.format == 'tabular':
            if isinstance(
                self.data.index, pd.core.indexes.datetimes.DatetimeIndex
            ):
                self._tags_list = list(self.data.columns)
            else:
                self._tags_list = list(self.data.columns)[1:]
            
    # ------------------------------------------------------
    # These methods add time series components to be plotted
    # ------------------------------------------------------
    def add_signal(self, signals_list):
        """
        This method will let you select which signals you want to plot. It will
        double check that the signals are, actually available in the tags list.
        This method will populate the ``signal_data`` property with the list of
        each dataframes containing the signals to plot.
        
        Parameters:
            signals_list (list of string):
                A list of tag names to be rendered when you call ``plot()``
                
        Raises:
            Exception: if some of the signals are not found in the tags list
        """
        if self._tags_list is None:
            self._build_tags_list()
            
        # Check the list of requested signals:
        intersection = [tag for tag in self._tags_list if tag in signals_list]
        if len(intersection) != len(signals_list):
            unknown_signals = [tag for tag in signals_list if tag not in intersection]
            raise Exception(f'Signals not found in the tags list: {unknown_signals}')
            
        self._signals_to_plot += signals_list
        
        # Extract the series and store them in the current object instance:
        for tag_name in signals_list:
            if self.verbose:
                print(f'Extracting {tag_name}')
            self._extract_series(tag_name)
        
        # Prepare the figure:
        self.fig_height = 4
        self.height_ratios = [8]
        self.nb_plots += 1
        
    def _extract_series(self, tag_name):
        """
        This private method extracts the time series data for a given tag and
        make it available as ``pandas.DataFrame``. Each timeseries will consist
        on a single column DataFrame with a column name equal to the tag name
        and the timestamp column is used as the DateTimeIndex. Resampling and
        forward fill will also happen in this method.
        
        Parameters:
            tag_name (string):
                The name of the tag to extract
        """
        if self._format == 'timeseries':
            df = self._extract_series_timeseries(tag_name)
            
        elif self._format == 'tabular':
            df = self._extract_series_tabular(tag_name)
            
        df = self._preprocess_timeseries(df)
        self._signals_data.append(df)
        del df

        start_date = np.min(self._signals_data[-1].index)
        end_date = np.max(self._signals_data[-1].index)
        
        if self._start_date is None:
            self._start_date = start_date
        elif self._start_date > start_date:
            self._start_date = start_date
            
        if self._end_date is None:
            self._end_date = end_date
        elif self._end_date < end_date:
            self._end_date = end_date
        
    def _extract_series_timeseries(self, tag_name):
        """
        Extracts all the data for the tag_name passed as an argument for dataset
        that follows the ``timeseries`` format.
        
        Parameters:
            tag_name (string):
                The name of the tag to extract
        """
        # Extract the data for the current signal:
        df = self._data[self._data[self.tag_col] == tag_name].copy()
        
        # Remove the tag name from the column list and use it as a name for
        # the last column remaining, that should contain the values:
        df = df.drop(columns=[self.tag_col])
        if self.timestamp_col is None:
            df.columns = [tag_name]
        else:
            df.columns = [self.timestamp_col, tag_name]
        
        return df
        
    def _extract_series_tabular(self, tag_name):
        """
        Extracts all the data for the tag_name passed as an argument for dataset
        that follows the ``tabular`` format.
        
        Parameters:
            tag_name (string):
                The name of the tag to extract
        """
        # Extract the data for the current signal:
        if self.timestamp_col is None:
            df = self._data[[tag_name]].copy()
        else:
            df = self._data[[self.timestamp_col, tag_name]].copy()
        
        return df

    def _preprocess_timeseries(self, df):
        """
        Preprocess the timeseries to add a timestamp index and resample the
        data for the tag dataframe passed as an argument
        
        Parameters:
            df (pandas.DataFrame):
                The dataframe that contains the tag data to preprocess
        """
        if self.timestamp_col is not None:
            # Convert the timestamp to a proper format (without timezone) and
            # use it as an index:
            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
            df[self.timestamp_col] = df[self.timestamp_col].dt.tz_localize(None)
            df = df.set_index(self.timestamp_col)

        # Resample the data if this was requested at initialization:
        if self.resample is not None:
            df = df.resample(self.resample).mean()
            df = df.fillna(method='ffill')
            
        return df

    # ----------------------------------------------------------------------
    # These methods add labels and predicted events components to be plotted
    # ----------------------------------------------------------------------
    def add_labels(self, labels_df, labels_title='Known anomalies'):
        """
        Add a label component to the plot to visualize the known anomalies
        periods as a secondary plot under the time series visualization panel.
        
        Parameters:
            labels_df (pandas.DataFrame):
                You can add one label ribbon, defined with a dataframe that 
                gives the start and end date of every known anomalies
            labels_title (string):
                Title to be used for the known anomalies label ribbon
        """
        self._labels_df = self._convert_ranges(labels_df)
        self._labels_title = labels_title
        self.nb_plots += 1
        self.fig_height += 1.0
        self.height_ratios += [1.5]
        
    def add_predictions(self, 
                        predictions_list,
                        prediction_titles=['Detected events']):
        """
        Add a prediction component to the plot to visualize detected events
        as a secondary plot under the time series visualization panel.
        
        Parameters:
            predictions_list (list of pandas.DataFrame):
                You can add several predictions ribbon. Each ribbon is defined
                with a dataframe that gives the start and end date of every
                detected events. Several ribbons can be grouped inside a list
            prediction_titles (list of strings):
                This lists contains all the titles to be used for each
                prediction ribbon
        """
        self._predictions_title = prediction_titles
        self.nb_plots += len(predictions_list)
        self.fig_height += 1.0 * len(predictions_list)
        self.height_ratios += [1.5] * len(predictions_list)
        self._predictions_ranges = predictions_list

        for df in predictions_list:
            self._predictions_df.append(self._convert_ranges(df))
    
    def _convert_ranges(self, ranges_df, default_freq='1min'):
        """
        This method expands a list of ranges into an datetime index 
        pandas.Series
        
        Parameters:
            ranges_df (pandas.DataFrame):
                A dataframe with two columns, the start and end timestamp of
                each event
            default_freq (string):
                The default frequency to generate the time range for. This will
                be used to generate the DateTimeIndex for this pandas.Series
                
        Returns:
            pandas.DataFrame: a dataframe with a DateTimeIndex spanning from the
            minimum to the maximum timestamps present in the input dataframe.
            This will be a single Series named "Label" where a value of 1.0
            will correspond to the presence of an event (labels or anomalies).
        """
        range_index = pd.date_range(
            start=self._start_date,
            end=self._end_date, 
            freq=default_freq
        )
        range_data = pd.DataFrame(index=range_index)
        range_data.loc[:, 'Label'] = 0.0

        for _, row in ranges_df.iterrows():
            event_start = row[0]
            event_end = row[1]
            range_data.loc[event_start:event_end, 'Label'] = 1.0
            
        return range_data
        
    # ------------------------------------------------------------
    # These methods add other features and decorations to the plot
    # ------------------------------------------------------------
    def add_train_test_split(self, 
                             split_timestamp, 
                             train_label='Train', 
                             test_label='Evaluation'):
        """
        Add a way to visualize the split between training and testing periods.
        The training period will stay colorful on the timeseries area of the
        plot while the testing period will be greyed out.
        
        Parameters:
            split_timestamp (string or datetime):
                The split date. If a string is passed, it will be converted
                into a datetime
            train_label (string):
                Name of the training period (will be visible in the legend)
            test_label (string):
                Name of the testing period (will be visible in the legend)
        """
        if type(split_timestamp) == str:
            split_timestamp = pd.to_datetime(split_timestamp)
            
        self._tag_split = split_timestamp
        self._split_labels = {'training': train_label, 'testing': test_label}
        
    def add_rolling_average(self, window_size):
        """
        Adds a rolling average over a time series plot
        
        Parameters:
            window_size (integer):
                Size of the window in time steps to average over
        """
        self._rolling_average = True
        self._rolling_average_window = window_size
        
    # -----------------------
    # Plot management methods
    # -----------------------
    def plot(self, fig_width=18, colors=DEFAULT_COLORS, labels_bottom=False, no_legend=False):
        """
        Renders the plot as configured with the previous function
        
        Parameters:
            fig_width (integer):
                The width of the figure to generate (defaults to 18)
                
        Returns:
            tuple: tuple containing:
                * A ``matplotlib.pyplot.figure`` where the plots are drawn
                * A ``list of matplotlib.pyplot.Axis`` with each plot drawn here
        """
        # Prepare the figure structure:
        self._set_plot_shared_params()
        fig = plt.figure(figsize=(fig_width, self.fig_height))
        gs = gridspec.GridSpec(nrows=self.nb_plots, 
                               ncols=1, 
                               height_ratios=self.height_ratios, 
                               hspace=0.5)
        ax = []
        for i in range(self.nb_plots):
            ax.append(fig.add_subplot(gs[i]))
            
        # First, we plot the time series signals:
        ax_id = 0
        for signal_df in self._signals_data:
            tag_name = signal_df.columns[0]
            
            # If we don't want to highlight training and evaluation period
            # we only plot the signal as is:
            if self._tag_split is None:
                ax[ax_id].plot(signal_df, alpha=0.8, label=tag_name)

            # Otherwise, we plot the training part in grey and the evaluation
            # part in color:
            else:
                self._plot_split_signal(signal_df, tag_name, ax[ax_id])
                
            # We can display a daily rolling average:
            if self._rolling_average:
                self._plot_rolling_average(signal_df, ax[ax_id])

        # Next, we plot the labels ranges, except if they should be at the bottom:
        if (self._labels_df is not None) and (labels_bottom == False):
            ax_id += 1
            self._plot_ranges(
                self._labels_df, 
                self._labels_title,
                colors['labels'],
                ax[ax_id]
            )

        # Next, we plot the detected event ranges:
        if len(self._predictions_df) > 0:
            for prediction_title, predictions_df in zip(self._predictions_title, self._predictions_df):
                ax_id += 1
                self._plot_ranges(
                    predictions_df, 
                    prediction_title, 
                    colors['predictions'],
                    ax[ax_id]
                )
                
        # Next, we plot the labels ranges, if they should be at the bottom:
        if (self._labels_df is not None) and (labels_bottom == True):
            ax_id += 1
            self._plot_ranges(
                self._labels_df, 
                self._labels_title,
                colors['labels'],
                ax[ax_id]
            )
                           
        # Legends:
        if not no_legend:
            ax[0].legend(**self._legend_format)
                               
        return fig, ax
            
    def _set_plot_shared_params(self):
        """
        Set shared parameters for all the plots to be rendered by this class
        """
        # plt.rcParams['figure.dpi'] = 300
        plt.rcParams['lines.linewidth'] = 0.5
        # plt.rcParams['axes.titlesize'] = 6
        # plt.rcParams['axes.labelsize'] = 6
        # plt.rcParams['xtick.labelsize'] = 4
        # plt.rcParams['ytick.labelsize'] = 4
        # plt.rcParams['grid.linewidth'] = 0.2
        plt.rcParams['legend.fontsize'] = 10
        
    def _plot_split_signal(self, signal_df, tag_name, ax):
        """
        Plot a timeseries signal in a provided ax with a different color for 
        training and testing periods
        
        Parameters:
            signal_df (pandas.DataFrame):
                The dataframe that will contain the timeseries data to plot
            tag_name (string):
                The tag name to use as a label
            ax (matplotlib.pyplot.Axis):
                The ax in which to render the time series plot
        """
        train_label = self._split_labels['training']
        ax.plot(signal_df[:self._tag_split],
                alpha=0.8, 
                label=f'{tag_name} - {train_label}')
                   
        test_label = self._split_labels['testing']
        ax.plot(signal_df[self._tag_split:],
                alpha=0.5, 
                color='tab:grey',
                label=f'{tag_name} - {test_label}')
                
    def _plot_ranges(self, range_df, range_title, color, ax):
        """
        Plot a range with either labelled or predicted events as a filled
        area positionned under the timeseries data.
        
        Parameters:
            range_df (pandas.DataFrame):
                A DataFrame that must contain at least a DateTimeIndex and a
                column called "Label"
            range_title (string):
                Title of the ax containing this range
            color (string):
                A string used as a color for the filled area of the plot
            ax (matplotlib.pyplot.Axis):
                The ax in which to render the range plot
        """
        ax.plot(range_df['Label'], color=color)
        ax.fill_between(range_df.index, 
                        y1=range_df['Label'], 
                        y2=0, 
                        alpha=0.1, 
                        color=color, 
                        label=range_title)
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        ax.set_xlabel(range_title, fontsize=12)
        
    def _plot_rolling_average(self, signal_df, ax):
        """
        Computes a rolling average of the signal passed as an argument and plot
        it in the given ax.
        
        Parameters:
            signal_df (pandas.DataFrame):
                The dataframe that will contain the timeseries to build the
                rolling average from. The rolling parameters will have to be
                already set with the ``add_rolling_average()`` method
            ax (matplotlib.pyplot.Axis):
                The ax in which to render the time series plot
        """
        daily_rolling_average = signal_df.iloc[:, 0].rolling(
            window=self._rolling_average_window
        ).mean()
        
        # Plot a thick white line around a red thin line for visibility:
        ax.plot(daily_rolling_average.index, 
                daily_rolling_average, 
                alpha=0.5, 
                color='white', 
                linewidth=3)
        ax.plot(daily_rolling_average.index, 
                daily_rolling_average, 
                label='Rolling leverage', 
                color='tab:red', 
                linewidth=1)
        
    # ----------
    # Histogram plot management functions
    # -----------
    def plot_histograms(self, freq='1min', prediction_index=0, top_n=8, fig_width=18, start=None, end=None):
        """
        Plot values distribution as histograms for the top contributing sensors.
        
        Parameters:
            freq (string):
                The datetime index frequence (defaults to '1min'). This must 
                be a string following this format: XXmin where XX is a number
                of minutes.
            prediction_index (integer):
                You can add several predicted ranges in your plot. Use this
                argument to specify for which one you wish to plot a histogram
                for (defaults to 0)
            top_n (integer):
                Number of top signals to plot (default: 8)
            fig_width (float):
                Width of the figure generated (default: 18)
            start (pandas.DatetTime):
                Start date of the range to build the values distribution for
                (default: None, use the evaluation period start)
            end (pandas.DatetTime):
                End date of the range to build the values distribution for
                (default: None, use the evaluation period end)
                
        Returns:
            matplotlib.pyplot.figure: a figure where the histograms are drawn
        """
        predicted_ranges_df = self._predictions_df[prediction_index]    
        abnormal_index = predicted_ranges_df[predicted_ranges_df['Label'] > 0.0].index
        normal_index = predicted_ranges_df[predicted_ranges_df['Label'] == 0.0].index
        expanded_results = self._build_feature_importance_dataframe(
            freq=freq, 
            prediction_index=prediction_index
        )
        
        if (start is not None) and (end is not None):
            abnormal_index = predicted_ranges_df[predicted_ranges_df['Label'] > 0.0]
            abnormal_index = abnormal_index.loc[start:end].index
            normal_index = predicted_ranges_df[predicted_ranges_df['Label'] == 0.0]
            normal_index = normal_index.loc[start:end].index
            expanded_results = expanded_results.loc[start:end]
        
        most_contributing_signals = list(expanded_results.sum().sort_values(ascending=False).head(top_n).index)
        most_contributing_signals = [tag.split('\\')[1] for tag in most_contributing_signals]
        
        num_tags = len(most_contributing_signals)
        num_rows = int(np.ceil(num_tags / 4))
        fig_height = 5.0 * num_rows * fig_width / 24.0
        fig = plt.figure(figsize=(fig_width,fig_height))

        for i, current_tag in enumerate(most_contributing_signals):
            ax = fig.add_subplot(num_rows, 4, i+1)
            current_df = self._data[current_tag]

            ts1 = current_df.reindex(normal_index).copy()
            ts2 = current_df.reindex(abnormal_index).copy()

            plot_histogram_comparison(ts2,
                                      ts1,
                                      ax=ax, 
                                      label_timeseries_1=f'Values during abnormal events',
                                      label_timeseries_2=f'Values during normal periods',
                                      num_bins=50)
            ax.set_title(current_tag)
            
        return fig
        
    def _build_feature_importance_dataframe(self, freq='1min', prediction_index=0):
        """
        Builds a feature importance dataframe with the importance evolution of 
        each signal over time.
        
        Parameters:
            freq (string):
                The datetime index frequence (defaults to '1min'). This must 
                be a string following this format: XXmin where XX is a number
                of minutes.
            prediction_index (integer):
                You can add several predicted ranges in your plot. Use this
                argument to specify for which one you wish to plot a histogram
                for (defaults to 0)
                
        Returns:
            pandas.DataFrame: a dataframe with the feature importance evolutio
            of each signal over time.
        """
        if self.expanded_results is None:
            expanded_results = []
            predicted_ranges = self._predictions_ranges[prediction_index]
            num_events = predicted_ranges.shape[0]
            for index, row in predicted_ranges.iterrows():
                new_row = dict()
                new_row.update({'start': row['start']})
                new_row.update({'end': row['end']})
                new_row.update({'prediction': 1.0})

                diagnostics = pd.DataFrame(row['diagnostics'])
                diagnostics = dict(zip(diagnostics['name'], diagnostics['value']))
                new_row = {**new_row, **diagnostics}

                expanded_results.append(new_row)

            expanded_results = pd.DataFrame(expanded_results)

            freq_int = int(freq[:-3])
            cols = list(expanded_results.columns)[3:]
            expanded_results['end2'] = expanded_results['end'] + pd.to_timedelta(freq_int, unit='m')

            df1 = expanded_results[['start'] + cols].copy()
            df2 = expanded_results[['end'] + cols].copy()
            df3 = expanded_results[['end2'] + cols].copy()

            df1.columns = ['timestamp'] + cols
            df2.columns = ['timestamp'] + cols
            df3.columns = ['timestamp'] + cols

            df3.iloc[:, 1:] = 0.0

            expanded_results = pd.concat([df1, df2, df3], axis='index').sort_index()
            expanded_results = expanded_results.sort_values(by='timestamp', ascending=True)
            expanded_results = expanded_results.set_index('timestamp')
            expanded_results = expanded_results.resample(rule=freq).first()
            expanded_results = expanded_results.fillna(method='ffill')
            
            self.expanded_results = expanded_results

        return self.expanded_results

    # -------
    # Getters
    # -------
    @property
    def data(self):
        """
        A ``pandas.DataFrame`` containing time series data to plot
        """
        return self._data
        
    @property
    def format(self):
        """
        Either ``timeseries`` or ``tabular`` depending on the format of your 
        time series.
        """
        return self._format
        
    @property
    def timestamp_col(self):
        """
        ``string`` specifying the name of the columns that contains the 
        timestamps
        """
        return self._timestamp_col
        
    @property
    def tag_col(self):
        """
        If `data_format` is ``timeseries``, this argument specifies what is the 
        name of the columns that contains the name of the tags
        """
        return self._tag_col
        
    @property
    def tags_list(self):
        """
        ``list of strings`` containing the list of all tags associated to the
        current dataset
        """
        if self._tags_list is None:
            self._build_tags_list()
            
        return self._tags_list

    @property
    def legend_format(self):
        """
        ``kwargs dict`` to configure the legend to be displayed when this
        class renders the requested plot
        """
        return self._legend_format
        
    @property
    def signal_data(self):
        """
        ``list of pandas.DataFrame`` containing the time series data to plot
        """
        return self._signals_data

    # -------
    # Setters
    # -------
    @legend_format.setter
    def legend_format(self, new_legend_format):
        self._legend_format = new_legend_format

def compute_bin_edges(signals, num_bins=10):
    """
    Computes aligned bin edges for all signals passed in argument
    
    Parameters:
        signals (array_like):
            An array holding the elements we want to compute histogram bins
            for. Could be two pandas.Series, numpy arrays, lists...
        num_bins (integer):
            Number of bins to compute (defaults to 10)
            
    Returns:
        list: a list of (num_bins + 1) edges that can be used to plot a
        histogram
    """
    # Checks if the argument is a nested type or a numeric one:
    if isinstance(signals[0], (int, float)):
        all_signals_min = np.min(signals)
        all_signals_max = np.max(signals)
        
    # For nested type (list of pandas.Series, list of lists...), we
    # need to compute the min and max of each component of the list:
    else:
        all_signals_max = None
        all_signals_min = None
        for s in signals:
            signal_max = np.max(s)
            if (all_signals_max is not None) and (signal_max > all_signals_max):
                all_signals_max = signal_max
            elif all_signals_max is None:
                all_signals_max = signal_max
                
            signal_min = np.min(s)
            if (all_signals_min is not None) and (signal_min < all_signals_min):
                all_signals_min = signal_min
            elif all_signals_min is None:
                all_signals_min = signal_min
        
    # Now we can compute the bin width and their edges:
    bin_width = (all_signals_max - all_signals_min)/num_bins
    bins = np.arange(
        all_signals_min, 
        all_signals_max + bin_width, 
        bin_width
    )
    
    return bins

def plot_histogram_comparison(timeseries_1, 
                              timeseries_2, 
                              ax=None, 
                              label_timeseries_1=None, 
                              label_timeseries_2=None, 
                              show_legend=True,
                              num_bins=10
                             ):
    """
    Takes two timeseries and plot a histogram showing their respective 
    distribution of values
    
    Parameters:
        timeseries_1 (array_like):
            The first time series to plot a histogram for
        timeseries_2 (array_like):
            The second time series to plot a histogram for
        ax (matplotlib.pyplot.Axis):
            The ax in which to render the range plot. If None, this
            function will create a figure and an ax. Default to 
            None
        label_timeseries_1 (string):
            The label for the first time series
        label_timeseries_2 (string):
            The label for the second time series
        show_legend (Boolean):
            True to display a legend on this histogram plot and False
            otherwise
        num_bins (integer):
            Number of bins to compute (defaults to 10)
    """
    bins = compute_bin_edges([timeseries_1, timeseries_2], num_bins=num_bins)
    
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
    
    ax.hist(timeseries_1, 
            density=True, 
            alpha=0.5, 
            bins=bins, 
            edgecolor='#FFFFFF', 
            color='tab:red', 
            label=label_timeseries_1)
            
    ax.hist(timeseries_2, 
            density=True, 
            alpha=0.5, 
            bins=bins, 
            edgecolor='#FFFFFF', 
            color='tab:blue', 
            label=label_timeseries_2)
    
    ax.grid(False)
    ax.get_yaxis().set_visible(False)
    
    if (show_legend) and \
       (label_timeseries_1 is not None) and \
       (label_timeseries_2 is not None):
        ax.legend(framealpha=0.5)
        
    return ax
    
def plot_event_barh(event_details, num_signals=10, fig_width=12):
    """
    Plot a horizontal bar chart with the feature importance of each signal
    that contributes to the event passed as an argument.
    
    Parameters:
        event_details (pandas.DataFrame):
            A dataframe with the sensor name and the feature importance in
            two columns
        num_signals (integer):
            States how many signals to plot in the bar chart (default to 10)
        fig_width (integer):
            Width of the figure to plot
    
    Returns:
        Returns:
            tuple: tuple containing:
                * A ``matplotlib.pyplot.figure`` where the plot is drawn
                * A ``matplotlib.pyplot.Axis`` where the plot is drawn
    """
    event_time = event_details.columns[1]
    num_values = event_details.shape[0]
    event_details.columns = ['name', 'value']
    event_details = event_details.sort_values(by='value')
    event_details_limited = event_details.tail(num_signals)
    
    # We can then plot a horizontal bar chart:
    y_pos = np.arange(event_details_limited.shape[0])
    values = list(event_details_limited['value'])
    
    fig = plt.figure(figsize=(fig_width, 0.6 * event_details_limited.shape[0]))
    ax = plt.subplot(1,1,1)
    ax.barh(y_pos, event_details_limited['value'], align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(event_details_limited['name'])
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    # Add the values in each bar:
    for i, v in enumerate(values):
        if v == 0:
            ax.text(0.0005, i, f'{v*100:.2f}%', color='#000000', verticalalignment='center')
        else:
            ax.text(0.0005, i, f'{v*100:.2f}%', color='#FFFFFF', fontweight='bold', verticalalignment='center')
            
    ax.vlines(x=1/num_values, ymin=-0.5, ymax=np.max(y_pos) + 0.5, linestyle='--', linewidth=2.0, color='#000000')
    ax.vlines(x=1/num_values, ymin=-0.5, ymax=np.max(y_pos) + 0.5, linewidth=4.0, alpha=0.3, color='#000000')

    plt.title(f'Event detected at {event_time}', fontsize=12, fontweight='bold')
    
    return fig, ax
    
def plot_range(range_df, range_title, color, ax, column_name):
    """
    Plot a range with either labelled or predicted events as a filled
    area positionned under the timeseries data.

    Parameters:
        range_df (pandas.DataFrame):
            A DataFrame that must contain at least a DateTimeIndex and a
            column called "Label"
        range_title (string):
            Title of the ax containing this range
        color (string):
            A string used as a color for the filled area of the plot
        ax (matplotlib.pyplot.Axis):
            The ax in which to render the range plot
        column_name (string):
            The column from the range_df dataframe to use to plot the range
    """
    ax.plot(range_df[column_name], color=color)
    ax.fill_between(range_df.index, 
                    y1=range_df[column_name], 
                    y2=0, 
                    alpha=0.1, 
                    color=color, 
                    label=range_title)
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])
    ax.set_xlabel(range_title, fontsize=12)