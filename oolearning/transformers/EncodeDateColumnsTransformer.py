import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar

from oolearning.transformers.TransformerBase import TransformerBase


class EncodeDateColumnsTransformer(TransformerBase):
    """
    Replaces each date column with numeric/boolean columns that represent things such as:
        year, month, day, hour, min, second, ..., is_weekday, is_weekend
    """
    def __init__(self, include_columns: list=None):
        """
        :param include_columns: list of encoded date columns to use (i.e. subset of `encoded_columns()`
        """
        super().__init__()

        if include_columns is not None:
            assert isinstance(include_columns, list)
            assert len(set(include_columns).difference(EncodeDateColumnsTransformer.encoded_columns())) == 0

        self._include_columns = include_columns

    @staticmethod
    def encoded_columns():
        return [
            'year',
            'month',
            'day',
            'hour',
            'minute',
            'second',
            'quarter',
            'week',
            'days_in_month',
            'day_of_year',
            'day_of_week',
            'is_leap_year',
            'is_month_end',
            'is_month_start',
            'is_quarter_end',
            'is_quarter_start',
            'is_year_end',
            'is_year_start',
            'is_us_federal_holiday',
            'is_weekday',
            'is_weekend',
        ]

    def _fit_definition(self, data_x: pd.DataFrame) -> dict:
        # no state to capture
        return {}

    @staticmethod
    def _get_date_columns(dataframe: pd.DataFrame):
            return dataframe.select_dtypes(include=[np.datetime64]).columns.values

    def _transform_definition(self, data_x: pd.DataFrame, state: dict) -> pd.DataFrame:
        date_columns = self._get_date_columns(dataframe=data_x)

        # no date_columns, nothing to modify
        if len(date_columns) == 0:
            return data_x

        # find the min/max dates used in the entire dataset; used to create the holiday calender
        min_date = None
        max_date = None
        for column in date_columns:
            temp_min = data_x[column].min()
            if min_date is None or temp_min < min_date:
                min_date = temp_min

            temp_max = data_x[column].max()
            if max_date is None or temp_max > max_date:
                max_date = temp_max

        calendar = USFederalHolidayCalendar()
        holidays = calendar.holidays(start=min_date, end=max_date).to_list()

        def create_date_columns(x):
            return pd.Series([x.year,
                              x.month,
                              x.day,
                              x.hour,
                              x.minute,
                              x.second,
                              x.quarter,
                              x.week,
                              x.days_in_month,
                              x.dayofyear,
                              x.dayofweek,
                              x.is_leap_year,
                              x.is_month_end,
                              x.is_month_start,
                              x.is_quarter_end,
                              x.is_quarter_start,
                              x.is_year_end,
                              x.is_year_start,

                              ])

        column_names_temp = [
            'year',
            'month',
            'day',
            'hour',
            'minute',
            'second',
            'quarter',
            'week',
            'days_in_month',
            'day_of_year',
            'day_of_week',
            'is_leap_year',
            'is_month_end',
            'is_month_start',
            'is_quarter_end',
            'is_quarter_start',
            'is_year_end',
            'is_year_start',
        ]

        for column in date_columns:
            temp = data_x[column].apply(create_date_columns)
            temp.index = data_x.index
            temp.columns = column_names_temp
            temp['is_us_federal_holiday'] = [pd.Timestamp(x).date() in holidays for x in
                                             data_x[column].values]
            temp['is_weekday'] = temp.day_of_week <= 4
            temp['is_weekend'] = temp.day_of_week >= 5

            # before we change the column names to include original column name,
            # we have to drop any of the columns not in self._include_columns
            if self._include_columns is not None:
                temp = temp[self._include_columns]

            # change column names to include original column name
            temp.columns = ['{}_{}'.format(column, x) for x in temp.columns.values]

            data_x = pd.concat(objs=[data_x.drop(columns=column), temp], axis=1)

        return data_x

    def peak(self, data_x: pd.DataFrame):
        pass
