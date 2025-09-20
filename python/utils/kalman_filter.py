import numpy as np


class KalmanFilter:
    def __init__(self, F, G, H, Q, R):
        """
        Initialize the Kalman Filter with the system dynamics and noise ml_models.
        :param F: State transition matrix
        :param G: Control input matrix
        :param H: Measurement matrix
        :param Q: Process noise covariance matrix
        :param R: Measurement noise covariance matrix
        """
        assert F.shape[0] == F.shape[1], "F matrix must be square."
        assert Q.shape[0] == Q.shape[1], "Q matrix must be square."
        assert F.shape[0] == G.shape[0], "State number and G matrix row number must be the same."
        assert F.shape[0] == H.shape[1], "State number and H matrix column number must be the same."
        assert F.shape[0] == Q.shape[0], "State number and Q matrix size must be the same."
        assert R.shape[0] == R.shape[1], "R matrix must be square."
        assert R.shape[0] == H.shape[0], "Measurement number and R matrix size must be the same."

        self.F = F
        self.G = G
        self.H = H
        self.Q = Q
        self.R = R
        self.state_number = F.shape[0]

        self.x_hat = np.zeros(3)
        self.u_prev = 0

        self.P = np.array([[3 ** 2, 0, 0],
                           [0, 300 ** 2, 0],
                           [0, 0, 0.2 ** 2]])

    def predict(self, x_hat, u, P):
        """
        Perform the prediction step of the Kalman filter.
        :param x_hat: Current state estimate
        :param u: Control input
        :param P: Current covariance estimate
        :return: Predicted state and covariance
        """
        x_hat_pred = self.F @ x_hat + self.G @ u
        P_pred = self.F @ P @ self.F.T + self.Q
        return x_hat_pred, P_pred

    def correct(self, x_hat_pred, y, P_pred):
        """
        Perform the correction step of the Kalman filter.
        :param x_hat_pred: Predicted state estimate
        :param y: Measurement
        :param P_pred: Predicted covariance
        :return: Updated state estimate and covariance
        """
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        x_hat = x_hat_pred + K @ (y - self.H @ x_hat_pred)
        P = (np.eye(self.state_number) - K @ self.H) @ P_pred
        return x_hat, P

    def update(self, u, z):
        if u != self.u_prev:
            self.P = np.array([[3 ** 2, 0, 0],
                               [0, 300 ** 2, 0],
                               [0, 0, 0.2 ** 2]])
            self.x_hat = np.zeros(3)
        self.u_prev = u

        self.x_hat, self.P = self.predict(self.x_hat, u, self.P)
        self.x_hat, self.P = self.correct(self.x_hat, z, self.P)
        return self.x_hat
