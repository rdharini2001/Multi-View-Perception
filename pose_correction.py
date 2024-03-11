import numpy as np

def transform_covariance(T, covariance):
    return np.dot(T, np.dot(covariance, T.T)) #T is the transformation (relative pose) between robot_A and robot_B

def reduce_covariance(robot_B_covariance, transformed_robot_A_covariance, alpha):
    reduced_covariance = robot_B_covariance - alpha * transformed_robot_A_covariance #alpha is user defined hyper-parameter that determines the magnitude of correction to be applied.
    return reduced_covariance
