import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict

def generate_noisy_target_values(feature_set: np.array, target_weights: np.array, noise_standard_deviation: float) -> np.array:
    outputs: list = []
    for feature_vector in feature_set:
        products = np.array(target_weights[1::])*feature_vector
        output = target_weights[0] + products.sum() + np.random.normal(0, 1)*noise_standard_deviation
        outputs.append(output)
    return np.array(outputs)


if __name__ == "__main__":
    feature_dimension: int = 3
    noise_standard_deviation: float = 0.0
    target_weights = np.array([0.1, 0.3, -0.5, 1.0])
    sample_sizes = list(range(feature_dimension+15, feature_dimension+116, 10))
    # for each sample size this should be done many times and the average and variance should be retained
    # one can then plot these quantities as a function of N
    for sample_size in sample_sizes:
        feature_set = np.random.normal(0, 1, (sample_size, feature_dimension))
        target_values = generate_noisy_target_values(feature_set, target_weights, noise_standard_deviation)
        regularization_parameter = 0.05/sample_size
        regressor = linear_model.Ridge(alpha=regularization_parameter)
        predictions_on_validation = cross_val_predict(regressor, feature_set, target_values, cv=sample_size)
        cross_validation_errors = (predictions_on_validation - target_values)**2
        cross_validation_error = 1/sample_size * cross_validation_errors.sum()


