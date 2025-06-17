import sys
import inspect
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list
from scipy.spatial.distance import squareform
import logging

logger = logging.getLogger(__name__)


def analyze_dsl_usage(functions):
    dsl_function_names = _get_dsl_function_names() + ['sorted', 'max', 'min']
    dsl_class_attributes = _get_dsl_class_attributes()
    dsl_primitives = [f'{function_name}(' for function_name in dsl_function_names] \
        + [f'.{attr}' for attr in dsl_class_attributes]
    dsl_usage = {name: 0 for name in dsl_primitives}
    correlation_matrix = np.zeros((len(dsl_primitives), len(dsl_primitives)), dtype=int)
    for code in functions:
        code_dsl_usage = []
        for primitive in dsl_primitives:
            if primitive in code:
                dsl_usage[primitive] += 1
                code_dsl_usage.append(primitive)
        for i, primitive in enumerate(code_dsl_usage):
            # correlation_matrix[dsl_primitives.index(primitive), dsl_primitives.index(primitive)] += 1 # worsens the visualization
            for j in range(i + 1, len(code_dsl_usage)):
                correlation_matrix[dsl_primitives.index(primitive), dsl_primitives.index(code_dsl_usage[j])] += 1
                correlation_matrix[dsl_primitives.index(code_dsl_usage[j]), dsl_primitives.index(primitive)] += 1
        
    # Display the correlation matrix
    plot_correlation_matrix(correlation_matrix, dsl_primitives)
    
    # create a pandas DataFrame for better visualization
    import pandas as pd
    df = pd.DataFrame.from_dict(dsl_usage, orient='index', columns=['Usage Count'])
    df = df.sort_values(by='Usage Count', ascending=False)
    return df


def plot_correlation_matrix(correlation_matrix, dsl_primitives):
    correlation_matrix, order = reorder_correlation_matrix(correlation_matrix)
    correlation_matrix = correlation_matrix.astype(float)
    correlation_matrix[correlation_matrix == 0] = np.nan  # Set zero values to NaN for better visualization
    dsl_primitives = [dsl_primitives[i] for i in order]

    plt.figure(figsize=(12, 10))
    plt.imshow(correlation_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.xticks(ticks=np.arange(len(dsl_primitives)), labels=dsl_primitives, rotation=90)
    plt.yticks(ticks=np.arange(len(dsl_primitives)), labels=dsl_primitives)
    plt.title('DSL Primitives Usage Correlation Matrix')
    plt.tight_layout()
    plt.show()


def reorder_correlation_matrix(corr_matrix):
    distance_matrix = 1 - corr_matrix/ np.max(corr_matrix)
    condensed_distance = squareform(distance_matrix, checks=False)
    linkage_matrix = linkage(condensed_distance, method="average")
    linkage_matrix = optimal_leaf_ordering(linkage_matrix, condensed_distance)
    ordered_indices = leaves_list(linkage_matrix)
    return corr_matrix[np.ix_(ordered_indices, ordered_indices)], ordered_indices


def _get_dsl_function_names(dsl_module_name='arc25.dsl'):
    dsl_function_names = [
        name for name, cls in inspect.getmembers(sys.modules[dsl_module_name], inspect.isfunction)
        if cls.__module__ == dsl_module_name
        and not name.startswith('_')
    ]
    logger.info(f'Found {len(dsl_function_names)} DSL functions in {dsl_module_name}')
    return dsl_function_names


def _get_dsl_function_classes(dsl_module_name='arc25.dsl'):
    dsl_function_names = [
        cls for name, cls in inspect.getmembers(sys.modules[dsl_module_name], inspect.isclass)
        if cls.__module__ == dsl_module_name
        and not name.startswith('_')
        and name != 'Img'
    ]
    logger.info(f'Found {len(dsl_function_names)} DSL classes in {dsl_module_name}')
    return dsl_function_names


def _get_dsl_class_attributes(dsl_module_name='arc25.dsl'):
    dsl_classes = _get_dsl_function_classes(dsl_module_name)
    attributes = {cls.__name__: [attribute for attribute in dir(cls) if not attribute.startswith('_')]
                  for cls in dsl_classes}
    unique_attributes = set(attr for attrs in attributes.values() for attr in attrs)
    logger.info(f'Found {len(unique_attributes)} unique attributes in DSL classes')
    return unique_attributes
