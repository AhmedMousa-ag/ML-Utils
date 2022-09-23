from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, LabelEncoder
import numpy as np
import pandas as pd


class process_data():
    def __init__(self, data, Y=None, automatic_procedure: bool = True, shuffle: bool = True):
        """
        Args:
            data: The data we want to process, currently it only process pandas data frame,
                     and if not pandas it will convert it to pandas data frame.
            Y: our labels that we don't want to process.
            automatic_procedure: accept only boolean, defines if you want to configure the process manually or automate it
            shuffle: accept only boolean, defines if you want to shuffle the data first or no shuffling
            """
        self.raw_data = self._convert_to_pandas(data)
        if shuffle:
            self.raw_data = self.shuffle_data(self.raw_data)

        self.prep_data = None
        self.Y = Y
        if automatic_procedure:
            self._initiat_preprocessing()

        # --------------

    def shuffle_data(self, data):
        return data.sample(frac=1).reset_index(drop=True)
        # --------------

    def encode_data(self, exclude: list = [], one_hot_lim_threshold: int = 10, data=None):
        """This methode will indicate string data and will one hot encode it except for what we tell it to exclude from the process.
        Args:
            exclude: list inputs contain the column names what we don't want to process.
            one_hot_lim_threshold: If number of unique values exceeds that limit it will automatically label_encode it,
                                 make 0 if you want to label all categories.
        return:
            prep_data: the one hot encoded data after preprocessing.
        """
        excluded = [self.Y]
        if len(exclude) > 0:
            excluded = pd.concat([excluded, exclude], axis=1)

        if data:
            prep_data = data
            if len(excluded) > 0:
                prep_data = self.prep_data.drop(excluded, axis=1)
            else:
                prep_data = self.prep_data
        else:
            prep_data = self.prep_data

        for col in prep_data.columns:
            if prep_data[col].dtypes == object:
                if len(prep_data[col].unique()) > one_hot_lim_threshold:
                    prep_data[col] = self._label_encode(prep_data[col])
                else:
                    encoded_data = self._onehotencode(prep_data[col])
                    prep_data = prep_data.drop(col, axis=1)
                    prep_data = pd.concat([prep_data, encoded_data], axis=1)

        if len(exclude) > 0:  # Checks if we wanted to exclude any columns
            prep_data[exclude] = self.raw_data[exclude]

        self.prep_data = prep_data
        return self.prep_data
        # ---------------

    def _onehotencode(self, data):
        encoder = OneHotEncoder(sparse=False)
        return self._convert_to_pandas(encoder.fit_transform(np.array(data).reshape(-1, 1)))
        # --------

    def _label_encode(self, data):
        encoder = LabelEncoder()
        return self._convert_to_pandas(encoder.fit_transform(data))
        # --------

    def scale_data(self, exclude: list = [], data=None, scaler: str = "MinMaxScaler"):
        if not scaler in ["StandardScaler", "MinMaxScaler"]:
            raise ValueError(f"{scaler} isn't a StandardScaler or MinMaxScaler," +
                             "please pick one.. or leave as the defualt MinMaxScaler")

        excluded = [self.Y]
        if len(exclude) > 0:
            excluded = pd.concat([excluded, exclude], axis=1)

        if data:
            prep_data = data
            self.Y = prep_data[excluded]
        else:
            if len(excluded) > 0:
                self.Y = self.raw_data[excluded]
                prep_data = self.raw_data.drop(excluded, axis=1)
            else:
                self.Y = self.raw_data[excluded]
                prep_data = self.raw_data

        for col in prep_data.columns:
            if prep_data[col].dtypes != object:
                if scaler == "StandardScaler":
                    prep_data[col] = self._standard_scaler(prep_data[col])
                else:
                    prep_data[col] = self._minmaxscaler(prep_data[col])
        self.prep_data = prep_data
        return self.prep_data

    def _standard_scaler(self, data):
        scaler = StandardScaler()
        return self._convert_to_pandas(scaler.fit_transform(np.array(data).reshape(-1, 1)))

    def _minmaxscaler(self, data):
        scaler = MinMaxScaler()
        return self._convert_to_pandas(scaler.fit_transform(np.array(data).reshape(-1, 1)))

    def _initiat_preprocessing(self):
        """This function automates the processes of ecnoding and scaling all data
        return: 
            self.prep_data: the final processed data
        """
        self.scale_data()
        self.encode_data()
        return self.prep_data

    def get_preprocessed_data(self):
        """This function return the preprocessed data"""
        return self._convert_to_pandas(self.prep_data), self.Y

    def _convert_to_pandas(self, data):
        if not isinstance(data, pd.DataFrame):
            return pd.DataFrame(data)
        else:
            return data


class y_encoder():
    def __init__(self, y, one_hot_encode: bool = True, automatic_process_data: bool = True):
        """This class encoder Y labels to zeros and ones
        Args:
            y: your data that you want to encode
            one_hot_encode: accept only boolean, defines which encoding methode we will use
            automatic_process_data: accept only bloolean, defines if you want to configure the process manually or automate it"""
        self.y = self._convert_to_pandas(y)
        self.one_hot_encode = one_hot_encode
        self.automatic_process_data = automatic_process_data
        if self.automatic_process_data:
            self.__iniate_process()
        # ---------

    def __iniate_process(self):
        self.encode_data()
        # --------

    def encode_data(self, data=None):
        if data:
            self.y = data

        if self.one_hot_encode:
            for col in self.y.columns:
                encoded_data = self._onehotencode(self.y[col])
                self.y = self.y.drop(col, axis=1)
                self.y = pd.concat([self.y, encoded_data], axis=1)
        else:
            for col in self.y.columns:
                encoded_data = self._label_encode(self.y[col])
                self.y = self.y.drop(col, axis=1)
                self.y = pd.concat([self.y, encoded_data], axis=1)
        # --------

    def _onehotencode(self, data):
        encoder = OneHotEncoder(sparse=False)
        return self._convert_to_pandas(encoder.fit_transform(np.array(data).reshape(-1, 1)))
        # --------

    def _label_encode(self, data):
        encoder = LabelEncoder()
        return self._convert_to_pandas(encoder.fit_transform(data))
        # --------

    def get_preprocessed_data(self):
        """This function return the preprocessed data"""
        return self._convert_to_pandas(self.y)

    def _convert_to_pandas(self, data):
        if not isinstance(data, pd.DataFrame):
            return pd.DataFrame(data)
        else:
            return data


def load_data(data_version, main_path):
    x_train = pd.read_csv(main_path + "/" + data_version + "/" + "x_train.csv")
    y_train = pd.read_csv(main_path + "/" + data_version + "/" + "y_train.csv")
    x_test = pd.read_csv(main_path + "/" + data_version + "/" + "x_test.csv")
    y_test = pd.read_csv(main_path + "/" + data_version + "/" + "y_test.csv")
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
