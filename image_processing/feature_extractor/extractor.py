import abc, six

@six.add_metaclass(abc.ABCMeta)
class FeatureExtractor():
    @abc.abstractmethod
    def extract_features(self, images, feature_df, regionprops):
        pass