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


def compute_average(sample: list) -> float:
    return np.array(sample).sum()/len(sample)

def compute_variance(sample: list) -> float:
    average = compute_average(sample)
    variance = 0
    for point in sample:
        variance += (point-average)**2
    return variance/len(sample)

def run_cross_validation_many_times(number_of_runs: int, sample_size: int, feature_dimension: int, target_weights: np.array, noise_standard_deviation: float) -> tuple:
    cross_validation_errors = []
    e_1_values = []
    e_2_values = []
    for run in range(number_of_runs):
        feature_set = np.random.normal(0, 1, (sample_size, feature_dimension))
        target_values = generate_noisy_target_values(feature_set, target_weights, noise_standard_deviation)
        regularization_parameter = 0.05/sample_size
        regressor = linear_model.Ridge(alpha=regularization_parameter)
        predictions_on_validation = cross_val_predict(regressor, feature_set, target_values, cv=sample_size)
        pointwise_cross_validation_errors = (predictions_on_validation - target_values)**2
        cross_validation_error = 1/sample_size * pointwise_cross_validation_errors.sum()
        cross_validation_errors.append(cross_validation_error)
        e_1_values.append(pointwise_cross_validation_errors[0])
        e_2_values.append(pointwise_cross_validation_errors[1])
    e_1_variance = compute_variance(e_1_values)
    average_e_1 = compute_average(e_1_values)
    e_2_variance = compute_variance(e_2_values)
    average_e_2 = compute_average(e_2_values)
    cross_validation_error_variance = compute_variance(cross_validation_errors)
    average_cross_validation_error = compute_average(cross_validation_errors)
    return (average_e_1, e_1_variance, average_e_2, e_2_variance, average_cross_validation_error, cross_validation_error_variance)

if __name__ == "__main__":
    number_of_runs = 1000
    feature_dimension: int = 3
    noise_standard_deviation: float = 0.5
    target_weights = np.array([0.1, 0.3, -0.5, 1.0])
    sample_sizes = list(range(feature_dimension+15, feature_dimension+116, 10))
    # for each sample size this should be done many times and the average and variance should be retained
    # one can then plot these quantities as a function of N
    for sample_size in sample_sizes:
        print(f"Running cross validation with sample size {sample_size}")
        average_e_1, e_1_variance, average_e_2, e_2_variance, average_cross_validation_error, cross_validation_error_variance = run_cross_validation_many_times(number_of_runs, sample_size, feature_dimension, target_weights, noise_standard_deviation)
        print("Average e_1")
        print(average_e_1)
        print("Var[e_1]")
        print(e_1_variance)
        print("Average e_2")
        print(average_e_2)
        print("Var[e_2]")
        print(e_2_variance)
        print("Average E_cv")
        print(average_cross_validation_error)
        print("Var[E_cv]")
        print(cross_validation_error_variance)
        # feature_set = np.random.normal(0, 1, (sample_size, feature_dimension))
        # target_values = generate_noisy_target_values(feature_set, target_weights, noise_standard_deviation)
        # regularization_parameter = 0.05/sample_size
        # regressor = linear_model.Ridge(alpha=regularization_parameter)
        # predictions_on_validation = cross_val_predict(regressor, feature_set, target_values, cv=sample_size)
        # cross_validation_errors = (predictions_on_validation - target_values)**2
        # cross_validation_error = 1/sample_size * cross_validation_errors.sum()


