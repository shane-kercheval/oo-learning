from typing import Union

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.decomposition import PCA

from oolearning.OOLearningHelpers import OOLearningHelpers
from oolearning.transformers.TransformerBase import TransformerBase


# noinspection PyTypeChecker, SpellCheckingInspection
class PCATransformer(TransformerBase):
    """
    Performs Principal Component Analysis.
    """
    def __init__(self,
                 percent_variance_explained: Union[float, None] = 0.95,
                 exclude_categorical_columns=False):
        """
        :param percent_variance_explained: "select the number of components such that the amount of variance
            that needs to be explained is greater than the percentage specified" by
            `percent_variance_explained`.
            http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

            Alternatively, the user can pass in `None`, which will give all components, and then use
                `plot_cumulative_variance()` to determine ideal number of components baed off of the bend in
                the graph.

        :param exclude_categorical_columns: if set to True, the categoric features are not retained in the
            transformed dataset returned.
        """
        super().__init__()
        self._percent_variance_explained = percent_variance_explained
        self._exclude_categorical_columns = exclude_categorical_columns
        self._cumulative_explained_variance = None
        self._component_explained_variance = None
        self._features = None
        self._number_of_components = None
        self._pca_object = None
        self._loadings = None

    def peak(self, data_x: pd.DataFrame):
        pass

    @property
    def component_explained_variance(self):
        return self._component_explained_variance

    @property
    def cumulative_explained_variance(self) -> np.array:
        return self._cumulative_explained_variance

    def loadings(self,
                 top_n_components: Union[int, None] = None,
                 top_n_features: Union[int, None] = None) -> pd.DataFrame:
        """
        returns the loading vectors for each compontent (columns are components, rows are features)
        :param top_n_components: only include the top n components. If `None`, use all components.
        :param top_n_features: only include the top n features for each component. If `None`, use all
            features.
        """
        loadings_df = pd.DataFrame(self._loadings,
                                   columns=self._features,
                                   index=['PC-'+str(x+1) for x in range(self._loadings.shape[0])]).transpose()

        if top_n_components is None:
            top_n_components = loadings_df.shape[1]

        loadings_df = loadings_df.iloc[:, 0:top_n_components]

        if top_n_features is None:
            return loadings_df
        else:
            return loadings_df[(loadings_df.abs().rank(ascending=False) <= top_n_features).any(axis=1)]

    def plot_loadings(self,
                      top_n_components: Union[int, None] = None,
                      top_n_features: Union[int, None] = None,
                      annotate: bool = True,
                      font_size: int = 7):
        """
        :param top_n_components: only include the top n components. If `None`, use all components.
        :param top_n_features: only include the top n features for each component. If `None`, use all
            features.
        :param annotate: whether or not to include the loading value within each cell
        :param font_size: the font size of the loading value within each cell
        """
        plt.title('PCA Loadings')
        sns.heatmap(self.loadings(top_n_components=top_n_components, top_n_features=top_n_features),
                    annot=annotate, annot_kws={"size": font_size}, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.gcf().tight_layout()

    def component_feature_ranking(self, ith_component: int, top_n: Union[int, None] = None) -> pd.Series:
        """

        :param ith_component: the index of the component (indexing starts at `1`)
        :param top_n: the number of top loadings to include (by absolute value of the loadings)
        :return: a sorted pandas.Series containing the top_n loadings of the ith_component, sorted by absolute
            value of the loadings of that component
        """
        if top_n is None:
            top_n = len(self.loadings())

        pca_column = 'PC-' + str(ith_component)
        df_copy = self.loadings().copy()
        df_copy['sort'] = df_copy[pca_column].abs()
        return df_copy.sort_values(by='sort', ascending=False).iloc[0:top_n][pca_column]

    @property
    def number_of_components(self) -> int:
        return self._number_of_components

    def _fit_definition(self, data_x: pd.DataFrame) -> dict:
        assert data_x.isna().sum().sum() == 0

        numeric_columns, categorical_features = OOLearningHelpers.get_columns_by_type(data_dtypes=data_x.dtypes,  # noqa
                                                                                      target_variable=None)
        # perform PCA on numeric features, then add on categorical features
        self._pca_object = PCA(n_components=self._percent_variance_explained, random_state=42)
        self._pca_object.fit(X=data_x.drop(columns=categorical_features))

        self._features = numeric_columns
        self._loadings = self._pca_object.components_
        self._component_explained_variance = self._pca_object.explained_variance_ratio_
        self._cumulative_explained_variance = np.cumsum(self._pca_object.explained_variance_ratio_)
        self._number_of_components = self._pca_object.n_components_

        return dict(categorical_features=categorical_features)

    def _transform_definition(self, data_x: pd.DataFrame, state: dict) -> pd.DataFrame:
        categorical_features = state['categorical_features']

        new_data = self._pca_object.transform(X=data_x.drop(columns=categorical_features))
        new_column_names = ['component_'+str(x + 1) for x in range(self._number_of_components)]
        transformed_data = pd.DataFrame(new_data, columns=new_column_names, index=data_x.index)
        assert data_x.shape[0] == transformed_data.shape[0]  # ensure same number of rows
        return transformed_data if self._exclude_categorical_columns else pd.concat([transformed_data,
                                                                                     data_x[categorical_features]],  # noqa
                                                                                    axis=1)

    def plot_cumulative_variance(self):
        """
        Creates a Pareto plot of PCA that shows the cumulative variance explained for each additional
            component used.
        """
        assert self._cumulative_explained_variance is not None

        cumulative_var = self.cumulative_explained_variance
        component_weights = np.array(
            [x - y for x, y in zip(cumulative_var, np.insert(cumulative_var, 0, 0, axis=0))])
        assert len(component_weights) == len(cumulative_var)

        # lefthand edge of each bar
        left = range(1, len(component_weights) + 1)
        fig, ax = plt.subplots(1, 1)
        ax.bar(left, component_weights, 1)
        ax.plot(range(1, len(cumulative_var) + 1), cumulative_var)
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.title('PCA Explained Variance')
        index_90_per = np.argmax(cumulative_var >= 0.90)
        index_95_per = np.argmax(cumulative_var >= 0.95)
        index_99_per = np.argmax(cumulative_var >= 0.99)

        # make sure indexes are unique
        annotation_indexes = list({index_90_per, index_95_per, index_99_per})

        for x in annotation_indexes:
            y = cumulative_var[x]
            component = x + 1
            plt.annotate('{} comps; {}%'.format(component, round(cumulative_var[x] * 100, 1)),
                         size=8,
                         xy=(component, y),
                         xytext=(component, y),
                         horizontalalignment='right', verticalalignment='bottom')

        plt.plot([x + 1 for x in annotation_indexes], [cumulative_var[x] for x in annotation_indexes], 'rx')
