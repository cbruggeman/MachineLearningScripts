from sklearn.pipeline import Pipeline

class FullPipeline(Pipeline):
    """Pipeline with estimator attributes.

    Quick and dirty extension of sklearn's pipeline. Allows for
    accessing attributes of the final estimator with the sytnax
    'pipe.name'. If 'name' is an attribute of the Pipeline 
    (eg. named_steps), then this will act as expected, otherwise it
    will get the attribute of the final estimator. For example,
    in classification, 'pipe.classes_' will return the 'classes_'
    attribute of the classifier at the end of the pipeline.

    This allows Pipelines to work with other aspects of sklearn
    such as BaggingClassifier, which were previously incompatible.

    This method could lead to some unintended consequences, and
    is intended only as a quick bandaid for the problem.
    """
    def __getattribute__(self,name):
        try:
            return Pipeline.__getattribute__(self,name)
        except AttributeError:
            return self.steps[-1][-1].__getattribute__(name)

