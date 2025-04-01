import os
import pickle
import pandas as pd
import json
from .test import Test


class TestSuite:
    def __init__(self):
        self.tests = []

    def add_test(self, test: Test):
        self.tests.append(test)

    def run_all(self, completion_model):
        for test in self.tests:
            test.run(completion_model)

    def summarize(self):
        results = []
        failure_count = 0
        total_tests = len(self.tests)

        for test in self.tests:
            result = test.summarize()
            if result['fail']:
                failure_count += 1
            results.append(result)

        fail_rate = (failure_count / total_tests) * 100 if total_tests > 0 else 0
        summary = {
            'total_tests': total_tests,
            'failures': failure_count,
            'fail_rate': fail_rate
        }

        return results, summary

    def export_results(self, filename, file_format='csv', overwrite=False):
        """
        Export the results of the test suite to a file in the specified format.

        Parameters:
        filename (str): The name of the file to export the results to.
        file_format (str): The format of the file ('csv', 'xlsx', or 'json').
        overwrite (bool): Whether to overwrite the file if it already exists.
        """
        if not overwrite and os.path.exists(filename):
            print(f"File {filename} already exists. Set overwrite=True to overwrite the file.")
            return

        results, _ = self.summarize()
        df = pd.DataFrame(results)

        if file_format == 'csv':
            df.to_csv(filename, index=False)
        elif file_format == 'xlsx':
            df.to_excel(filename, index=False)
        elif file_format == 'json':
            with open(filename, 'w') as json_file:
                json.dump(results, json_file, indent=4)
        else:
            print(f"Unsupported file format: {file_format}")

    def clear(self):
        self.tests = []

    def save(self, filename, overwrite=False):
        if not overwrite and os.path.exists(filename):
            print(f"File {filename} already exists. Set overwrite=True to overwrite the file.")
            return

        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
