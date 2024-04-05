import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import namedtuple
from rdt.transformers import ClusterBasedNormalizer , OneHotEncoder

SpanInfo = namedtuple('SpanInfo', ['dim', 'activation_fn'])
ColumnTransformInfo = namedtuple(
    'ColumnTransformInfo', [
        'column_name', 'column_type', 'transform', 'output_info', 'output_dimensions'
    ]
)




class DataTransformer(object):
    def __init__(self, rawdata, cat_cols, num_cols,mixed_cols, log_cols, target_col, max_clusters=10, weight_threshold=0.005):   
       self.rawdata = rawdata   #type:Dataframe
       self.cat_cols = cat_cols  #type:list
       self.num_cols = num_cols   #type:list
       self.mixed_cols = mixed_cols #type: dictionary:keys(name of mixed cols), values(list of categorical values)
       self.log_cols = log_cols    #list of skewed exponential numerical columns
       self.target_col = target_col

       self._max_clusters = max_clusters
       self._weight_threshold = weight_threshold

    def transformData(self):
        # Spliting the input data to obtain training dataset
        raw_df = self.rawdata
        y_real = raw_df[self.target_col]
        X_real = raw_df.drop(columns=[self.target_col])
        X_train_real, _, y_train_real, _ = train_test_split(X_real ,y_real, test_size= 0.1, stratify=y_real,random_state=42)
        X_train_real[self.target_col]= y_train_real

        # Replacing empty strings with na if any and replace na with empty
        df = X_train_real
        df = df.replace(r' ', np.nan)
        df = df.fillna('empty')

        # Dealing with empty values in numeric columns by replacing it with -9999999 and treating it as categorical mode 
        all_columns= set(df.columns)
        irrelevant_missing_columns = set(self.cat_cols)
        relevant_missing_columns = list(all_columns - irrelevant_missing_columns)

        for i in relevant_missing_columns:
            if i in list(self.mixed_cols.keys()):
                if "empty" in list(df[i].values):
                    df[i] = df[i].apply(lambda x: -9999999 if x=="empty" else x )
                    self.mixed_cols[i].append(-9999999)
            else:
                if "empty" in list(df[i].values):   
                    df[i] = df[i].apply(lambda x: -9999999 if x=="empty" else x)
                    self.mixed_cols[i] = [-9999999]

        lower_bounds = {}
        # Dealing with skewed exponential numeric distributions by applying log transformation
        if self.log_cols:
            for log_column in log_columns:
                # Value added to apply log to non-positive numeric values
                eps = 1 
                # Missing values indicated with -9999999 are skipped
                lower = np.min(df.loc[df[log_column]!=-9999999][log_column].values) 
                lower_bounds[log_column] = lower
                if lower>0: 
                    df[log_column] = df[log_column].apply(lambda x: np.log(x) if x!=-9999999 else -9999999)
                elif lower == 0:
                    df[log_column] = df[log_column].apply(lambda x: np.log(x+eps) if x!=-9999999 else -9999999) 
                else:
                    # Negative values are scaled to become positive to apply log
                    df[log_column] = df[log_column].apply(lambda x: np.log(x-lower+eps) if x!=-9999999 else -9999999)
        return df

    def _fit_continuous(self, data):
        """Train Bayesian GMM for continuous columns.
        Args:
            data (pd.DataFrame):
                A dataframe containing a column.
        Returns:
            namedtuple:
                A ``ColumnTransformInfo`` object.
        """
        column_name = data.columns[0]
        gm = ClusterBasedNormalizer(max_clusters=min(len(data), self._max_clusters ))
        gm.fit(data, column_name)
        num_components = sum(gm.valid_component_indicator)

        return ColumnTransformInfo(
            column_name=column_name, column_type='continuous', transform=gm,
            output_info=[SpanInfo(1, 'tanh'), SpanInfo(num_components, 'softmax')],
            output_dimensions=1 + num_components)

    def _fit_discrete(self, data):
        """Fit one hot encoder for discrete column.
        Args:
            data (pd.DataFrame):
                A dataframe containing a column.
        Returns:
            namedtuple:
                A ``ColumnTransformInfo`` object.
        """
        column_name = data.columns[0]
        ohe = OneHotEncoder()
        ohe.fit(data, column_name)
        num_categories = len(ohe.dummies)

        return ColumnTransformInfo(
            column_name=column_name, column_type='discrete', transform=ohe,
            output_info=[SpanInfo(num_categories, 'softmax')],
            output_dimensions=num_categories)

    def _fit_mixed(self, data):
        """Train Bayesian GMM for mixed columns.
        Args:
            data (pd.DataFrame):
                A dataframe containing a column.
        Returns:
            namedtuple:
                A ``ColumnTransformInfo`` object.
        """
        column_name = data.columns[0]
        gm1 = ClusterBasedNormalizer(max_clusters=min(len(data), self._max_clusters))
        gm2 = ClusterBasedNormalizer(max_clusters=min(len(data), self._max_clusters))
        # first bgm model is fit to the entire data only for the purposes of obtaining a normalized value of any particular categorical mode
        gm1.fit(data, column_name)
        # main bgm model used to fit the continuous component and serves the same purpose as with purely numeric columns
        data_num = data.copy()
        data_num = data_num[~data_num[column_name].isin(self.mixed_cols[column_name])]
        gm2.fit(data_num, column_name)
        num_components = sum(gm2.valid_component_indicator) + len(self.mixed_cols[column_name])
        
        return ColumnTransformInfo(
            column_name=column_name, column_type='mixed', transform=(gm1, gm2),
            output_info=[SpanInfo(1, 'tanh'), SpanInfo(num_components, 'softmax')],
            output_dimensions = 1 + num_components)
    
    def fit(self, df):
        """Fit the ``DataTransformer``.
        Fits a ``ClusterBasedNormalizer`` for continuous columns and mixed columns, and a
        ``OneHotEncoder`` for discrete columns.
        This step also counts the #columns in matrix data and span information.
        """
        self.output_info_list = []
        self.output_dimensions = 0
        self.dataframe = True

        self._column_raw_dtypes = df.infer_objects().dtypes
        self._column_transform_info_list = []
        for column_name in df.columns:
            if column_name in self.cat_cols:
                column_transform_info = self._fit_discrete(df[[column_name]])
            elif column_name in self.num_cols:
                column_transform_info = self._fit_continuous(df[[column_name]])
            else: 
                column_transform_info = self._fit_mixed(df[[column_name]])

            self.output_info_list.append(column_transform_info.output_info)
            self.output_dimensions += column_transform_info.output_dimensions
            self._column_transform_info_list.append(column_transform_info)
                
    # convert a continuous column of data to appropriate format for discriminator
    def _transform_continuous(self, column_transform_info, data):
        column_name = data.columns[0]
        flattened_column = data[column_name].to_numpy().flatten()
        data = data.assign(**{column_name: flattened_column})
        gm = column_transform_info.transform
        transformed = gm.transform(data)

        #  Converts the transformed data to the appropriate output format.
        #  The first column (ending in '.normalized') stays the same,
        #  but the lable encoded column (ending in '.component') is one hot encoded.
        output = np.zeros((len(transformed), column_transform_info.output_dimensions))
        output[:, 0] = transformed[f'{column_name}.normalized'].to_numpy()
        index = transformed[f'{column_name}.component'].to_numpy().astype(int)
        output[np.arange(index.size), index + 1] = 1.0

        return output

    # convert a discrete column of data to appropriate format for discriminator
    def _transform_discrete(self, column_transform_info, data):
        ohe = column_transform_info.transform
        return ohe.transform(data).to_numpy()

    def _transform_mixed(self, column_transform_info, data):
        column_name = data.columns[0]
        output = np.zeros((len(data), column_transform_info.output_dimensions))
        #list of categorical values
        modal = self.mixed_cols[column_name]
        data_reset = data.reset_index()
        #extract the rows containing categorical values and create one-hot encoding for categorical values
        data_cat = data_reset[data_reset[column_name].isin(self.mixed_cols[column_name])]
        one_hot_df = pd.get_dummies(data_cat[column_name])
        #extract the rows containing numerical values
        data_num = data_reset[~data_reset[column_name].isin(self.mixed_cols[column_name])]
        #index of the empty rows
        data_empty_rows = data_reset[data_reset[column_name] == -9999999]

        #transform categorical values
        gm1 = column_transform_info.transform[0]
        transformed1 = gm1.transform(data_cat[[column_name]])
        result = pd.concat([transformed1[[f'{column_name}.normalized']], one_hot_df ], axis=1)
        output[list(result.index), : result.shape[1]] = result.values
        output[list(data_empty_rows.index), 0] = 0

        #transform numerical values
        gm2 = column_transform_info.transform[1]
        transformed2 = gm2.transform(data_num[[column_name]])
        output[list(transformed2.index), 0]= transformed2[f'{column_name}.normalized'].to_numpy()

        index = transformed2[f'{column_name}.component'].to_numpy().astype(int)
        output[list(transformed2.index), index + 1 + one_hot_df.shape[1]] = 1.0

        return output

    def _synchronous_transform(self, df, column_transform_info_list):
        """Take a Pandas DataFrame and transform columns synchronous.
        Outputs a list with Numpy arrays.
        """
        column_data_list = []
        for column_transform_info in column_transform_info_list:
            column_name = column_transform_info.column_name
            data = df[[column_name]]
            if column_transform_info.column_type == 'continuous':
                column_data_list.append(self._transform_continuous(column_transform_info, data))
            elif column_transform_info.column_type == 'discrete':
                column_data_list.append(self._transform_discrete(column_transform_info, data))
            else: 
                column_data_list.append(self._transform_mixed(column_transform_info, data))

        return column_data_list

    def transform(self, df):
        """Take raw data and output a matrix data."""
        column_data_list = self._synchronous_transform(df,self._column_transform_info_list)
       
        return np.concatenate(column_data_list, axis=1).astype(float)

    def _inverse_transform_continuous(self, column_transform_info, column_data, sigmas, st):
        gm = column_transform_info.transform
        data = pd.DataFrame(column_data[:, :2], columns=list(gm.get_output_sdtypes()))
        data[data.columns[1]] = np.argmax(column_data[:, 1:], axis=1)
        if sigmas is not None:
            selected_normalized_value = np.random.normal(data.iloc[:, 0], sigmas[st])
            data.iloc[:, 0] = selected_normalized_value

        return gm.reverse_transform(data)

    def _inverse_transform_discrete(self, column_transform_info, column_data):
        ohe = column_transform_info.transform
        data = pd.DataFrame(column_data, columns=list(ohe.get_output_sdtypes()))
        return ohe.reverse_transform(data)[column_transform_info.column_name]

    def _inverse_transform_mixed(self, column_transform_info, column_data):
        gm2 = column_transform_info.transform[1]
        column_name = column_transform_info.column_name
        extra_bits_len = len(self.mixed_cols[column_name])
        extra_bits = column_data[:, 1: 1 + extra_bits_len ]
        # find the indexes of one-hot-encoded and all-zero arrays
        one_hot_indexes = np.where(np.sum(extra_bits, axis=1) == 1)[0]
        all_zero_indexes = np.where(np.sum(extra_bits, axis=1) == 0)[0]
        #which index of the one-hot encoded elements of extra-bits is one. 
        one_indexes = np.argmax(extra_bits[one_hot_indexes], axis=1)
        mapping = self.mixed_cols[column_name]    #array of categorical values
        #substituted is the true values of categorical values
        substituted = np.choose(one_indexes, mapping)
        #the values and their indexes in the original data (for categorical ones)
        arr1 = substituted
        ind1 = one_hot_indexes
        data = pd.DataFrame(column_data[all_zero_indexes, :2], columns=list(gm2.get_output_sdtypes()))
        data[data.columns[1]] = np.argmax(column_data[all_zero_indexes, 1 + extra_bits_len:], axis=1)
        df2 = gm2.reverse_transform(data)
        arr2 = df2.values.reshape(-1)
        ind2 = all_zero_indexes
        merged_arr = np.empty(len(ind1) + len(ind2), dtype=arr1.dtype)
        merged_arr[ind1] = arr1
        merged_arr[ind2] = arr2

        # create a pandas dataframe with one column from the merged array
        df = pd.DataFrame({column_name: merged_arr})
        return df

    def inverse_transform(self, data, sigmas=None):
        """Take matrix data and output raw data.
        Output uses the same type as input to the transform function.
        Either np array or pd dataframe.
        st = 0 """
        st = 0
        sigmas=None 
        recovered_column_data_list = []
        column_names = []
        for column_transform_info in self._column_transform_info_list:
            dim = column_transform_info.output_dimensions
            column_data = data[:, st:st + dim]
            if column_transform_info.column_type == 'continuous':
                recovered_column_data = self._inverse_transform_continuous(column_transform_info, column_data, sigmas, st)
            elif column_transform_info.column_type == 'discrete':
                recovered_column_data = self._inverse_transform_discrete(column_transform_info, column_data)
            elif column_transform_info.column_type == 'mixed':
                recovered_column_data = self._inverse_transform_mixed(column_transform_info, column_data)
            recovered_column_data_list.append(recovered_column_data)
            column_names.append(column_transform_info.column_name)
            st += dim

        recovered_data = np.column_stack(recovered_column_data_list)
        recovered_data = (pd.DataFrame(recovered_data, columns=column_names).astype(self._column_raw_dtypes))
        if not self.dataframe:
            recovered_data = recovered_data.to_numpy()

        return recovered_data