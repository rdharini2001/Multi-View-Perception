import numpy as np

def transform_covariance(T, covariance):
    return np.dot(T, np.dot(covariance, T.T))

def reduce_covariance(robot_B_covariance, transformed_robot_A_covariance, alpha):
    reduced_covariance = robot_B_covariance - alpha * transformed_robot_A_covariance
    return reduced_covariance
