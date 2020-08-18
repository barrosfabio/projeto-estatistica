from hierarchical_classifier.classification_pipeline.classification_pipeline import HierarchicalClassificationPipeline

train_path = 'C:/Users/Fabio Barros/Git/projeto-estatistica/feature_extraction/result/covid_feature_matrix_train.csv'
test_path = 'C:/Users/Fabio Barros/Git/projeto-estatistica/feature_extraction/result/covid_feature_matrix_test.csv'
resampling_algorithm = 'smote'
classifier_name = 'rf'
resampling_strategy = 'flat'


classification_pipeline = HierarchicalClassificationPipeline(train_path, test_path, resampling_algorithm, classifier_name,
                                                             resampling_strategy)

classification_pipeline.run()