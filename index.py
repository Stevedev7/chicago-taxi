from kfp.compiler import Compiler
from kfp.components import load_component_from_file
from kfp.dsl import pipeline

@pipeline(name='Chicago-taxi-dataset')
def taxi_pipeline():
    get_data = load_component_from_file('components/get-data/get-data.yml')
    data_preparation = load_component_from_file('components/data-preparation/data-preparation.yml')
    preprocessing = load_component_from_file('components/preprocessing/preprocessing.yml')
    feature_scale = load_component_from_file('components/feature-scale/feature-scale.yml')
    train = load_component_from_file('components/neural-network/train/train.yml')
    test = load_component_from_file('components/neural-network/test/test.yml')

    get_data_task = get_data()
    # get_data_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
    data_preparation_task = data_preparation(get_data_task.outputs["Dataset"])
    preprocessing_task = preprocessing(data_preparation_task.outputs['X'], data_preparation_task.outputs['y'])
    # preprocessing_task.execution_options.caching_strategy.max_cache_staleness = "P0D" # Disable cache
    feature_scaling_task = feature_scale(preprocessing_task.outputs['X Train'], preprocessing_task.outputs['X Test'])
    # feature_scaling_task.execution_options.caching_strategy.max_cache_staleness = "P0D"
    train_task = train(feature_scaling_task.outputs['X Train Feature Scaled'], preprocessing_task.outputs['Y Train'])
    test_task = test(train_task.outputs["Model"], preprocessing_task.outputs["Y Test"], feature_scaling_task.outputs["X Test Feature Scaled"])
    test_task.execution_options.caching_strategy.max_cache_staleness = "P0D"

Compiler().compile(taxi_pipeline, 'pipeline.yml')