import subprocess
import sys
import time
from py4j.java_gateway import JavaGateway
import pandas as pd
import os


class JavaLinearRegression:
    def __init__(self, jar_path="libs/py4j-0.10.9.jar", class_dir="bin", entry_class="LinearRegressionEntry"):
        # Base directory = location of lrj.py
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Build absolute paths
        jar_path = os.path.join(base_dir, jar_path)
        class_dir = os.path.join(base_dir, class_dir)

        # Detect platform for classpath separator
        sep = ";" if sys.platform.startswith("win") else ":"
        cp = f"{class_dir}{sep}{jar_path}"

        # Start Java gateway process
        self.process = subprocess.Popen(
            ["java", "-cp", cp, entry_class],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Wait for Java to boot up
        time.sleep(2)

        # Connect to gateway
        self.gateway = JavaGateway()
        self.lr = self.gateway.entry_point.getLinearRegression()

    def _to_java_array(self, values):
        """Convert list/Series/DataFrame column into Java double[]"""
        if isinstance(values, pd.DataFrame):
            if values.shape[1] != 1:
                raise ValueError("DataFrame must have exactly one column for simple regression")
            values = values.iloc[:, 0].values
        elif isinstance(values, pd.Series):
            values = values.values

        X_java = self.gateway.new_array(self.gateway.jvm.double, len(values))
        for i, val in enumerate(values):
            X_java[i] = float(val)
        return X_java

    def fit(self, X, Y):
        X_java = self._to_java_array(X)
        Y_java = self._to_java_array(Y)
        self.lr.fit(X_java, Y_java)

    def get_slope(self):
        return self.lr.getSlope()

    def get_intercept(self):
        return self.lr.getIntercept()

    def predict(self, x):
        # x can be float, list, Series, or DataFrame column
        if isinstance(x, (list, pd.Series)):
            return [self.lr.predict(float(v)) for v in x]
        elif isinstance(x, pd.DataFrame):
            if x.shape[1] != 1:
                raise ValueError("DataFrame must have exactly one column for prediction")
            return [self.lr.predict(float(v)) for v in x.iloc[:, 0].values]
        else:
            return self.lr.predict(float(x))

    def close(self):
        self.gateway.shutdown()
        self.process.terminate()
