import pandas as pd
import pickle
from PromptOps.perturb import Perturbation
class ShotTemplateFormatter:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = self.load_data()
        self.perturb = Perturbation()  # Initialize the perturbation class

    def load_data(self):
        """
        Load the dataset based on the file extension.
        """
        if self.filepath.endswith('.csv'):
            return pd.read_csv(self.filepath)
        elif self.filepath.endswith('.xlsx'):
            return pd.read_excel(self.filepath)
        elif self.filepath.endswith('.json'):
            return pd.read_json(self.filepath)
        elif self.filepath.endswith('.pkl') or self.filepath.endswith('.pickle'):
            return self.load_pickle_file(self.filepath)
        else:
            raise ValueError("Unsupported file format")
    
    def load_pickle_file(self, filepath):
        """
        Load a dataset from a pickle file.
        """
        try:
            with open(filepath, 'rb') as file:
                data = pickle.load(file)
            return pd.DataFrame(data)  # Convert the loaded data to a DataFrame if applicable
        except Exception as e:
            raise ValueError(f"Error loading pickle file: {e}")
    def perturb_question(self, question, perturb_type):
        """
        Apply perturbation to the question based on the perturbation type.
        """
        if perturb_type == 'taxonomy':
            return self.perturb.taxonomy(question)
        elif perturb_type == 'negation':
            return self.perturb.negation(question)
        elif perturb_type == 'coreference':
            return self.perturb.coreference(question, 'word')  # Specify word to clarify
        elif perturb_type == 'srl':
            return self.perturb.srl(question)
        elif perturb_type == 'logic':
            return self.perturb.logic(question)
        elif perturb_type == 'fairness':
            return self.perturb.fairness(question)
        elif perturb_type == 'temporal':
            return self.perturb.temporal(question)
        elif perturb_type == 'ner':
            return self.perturb.ner(question)
        elif perturb_type == 'vocab':
            return self.perturb.vocab(question)
        else:
            raise ValueError("Invalid perturbation type")

    def format_zero_shot(self, row, perturb_type=None):
        """
        Format zero-shot template with optional perturbation for the main question.
        """

        if perturb_type == 'robust':
            # original = f"Q: {row['Original_Question']}\nA:"
            # perturbed = f"Q: {row['Perturbed_Question']}\nA:"
            original = f"{row['Prefix']}\n{row['Original_Question']}"
            perturbed = f"{row['Prefix']}\n{row['Perturbed_Question']}"
            return {
                "Original_Question_Index": row['Original_Question_Index'],
                "original_prompt": original,
                "perturb_prompt": perturbed,
                "perturb_type": perturb_type,
                "Perturbation": row['Perturbation'],
                "expected_result": row['Expected_Answer']
            }
        else:
            question = row['Question']
            if perturb_type:
                question = self.perturb_question(question, perturb_type)
            elif 'Perturbed' in row and row['Perturbed']:
                question = row['Perturbed']
            # original = f"Q: {row['Question']}\nA:"
            # perturbed = f"Q: {question}\nA:"
            original = f"{row['Prefix']}\n{row['Question']}"
            perturbed = f"{row['Prefix']}\n{question}"
            return {
                "original_prompt": original,
                "perturb_prompt": perturbed,
                "perturb_type": perturb_type,
                "expected_result": row['Expected_answer']
            }

    def format_one_shot(self, row, perturb_type=None):
        """
        Format one-shot template with optional perturbation for the main question.
        """
        if perturb_type == 'robust':
            # original = f"Q: {row['Question_1']}\nA: {row['Answer_1']}\n\nQ: {row['Original_Question']}\nA:"
            # perturbed = f"Q: {row['Question_1']}\nA: {row['Answer_1']}\n\nQ: {row['Perturbed_Question']}\nA:"
            original = f"{row['Prefix']}\n{row['Original_Question']}"
            perturbed = f"{row['Prefix']}\n{row['Perturbed_Question']}"
            return {
                "Original_Question_Index": row['Original_Question_Index'],
                "original_prompt": original,
                "perturb_prompt": perturbed,
                "perturb_type": perturb_type,
                "Perturbation": row['Perturbation'],
                "expected_result": row['Expected_Answer']
            }
        else:
            question = row['Question']
            if perturb_type:
                question = self.perturb_question(question, perturb_type)
            # original = f"Q: {row['Question_1']}\nA: {row['Answer_1']}\n\nQ: {row['Question']}\nA:"
            # perturbed = f"Q: {row['Question_1']}\nA: {row['Answer_1']}\n\nQ: {question}\nA:"

            original = f"{row['Prefix']}\n{row['Question']}"
            perturbed = f"{row['Prefix']}\n{question}"

            return {
                "original_prompt": original,
                "perturb_prompt": perturbed,
                "perturb_type": perturb_type,
                "expected_result": row['Expected_answer']
            }

    def format_few_shot(self, row, perturb_type=None):
        """
        Format few-shot template with optional perturbation for the main question.
        """
        

        if perturb_type == 'robust':
            question = row['Perturbed_Question']
            # few_shot_context = ""
            # for i in range(1, len(row)//2):  # Assuming paired Question_X and Answer_X columns
            #     if f'Question_{i}' in row and f'Answer_{i}' in row:
            #         few_shot_context += f"Q: {row[f'Question_{i}']}\nA: {row[f'Answer_{i}']}\n\n"

            # original = f"{few_shot_context}Q: {row['Original_Question']}\nA:"
            # perturbed = f"{few_shot_context}Q: {question}\nA:"
            original = f"{row['Prefix']}\n{row['Original_Question']}"
            perturbed = f"{row['Prefix']}\n{row['Perturbed_Question']}"
            return {
                "Original_Question_Index": row['Original_Question_Index'],
                "original_prompt": original,
                "perturb_prompt": perturbed,
                "perturb_type": perturb_type,
                "Perturbation": row['Perturbation'],
                "expected_result": row['Expected_Answer']
            }
        else:
            question = row['Question']
            if perturb_type:
                question = self.perturb_question(question, perturb_type)

            # few_shot_context = ""
            # for i in range(1, len(row)//2):  # Assuming paired Question_X and Answer_X columns
            #     if f'Question_{i}' in row and f'Answer_{i}' in row:
            #         few_shot_context += f"Q: {row[f'Question_{i}']}\nA: {row[f'Answer_{i}']}\n\n"

            # original = f"{few_shot_context}Q: {row['Original_Question']}\nA:"
            # perturbed = f"{few_shot_context}Q: {question}\nA:"
            original = f"{row['Prefix']}\n{row['Question']}"
            perturbed = f"{row['Prefix']}\n{question}"
            return {
                "original_prompt": original,
                "perturb_prompt": perturbed,
                "perturb_type": perturb_type,
                "expected_result": row['Expected_answer']
            }

    def format_all_rows(self, shot_type='zero', perturb_type=None):
        """
        Format all rows based on the specified shot type and apply perturbations if specified.
        """
        formatted_data = []

        for _, row in self.df.iterrows():
            if shot_type == 'zero':
                formatted_data.append(self.format_zero_shot(row, perturb_type=perturb_type))
            elif shot_type == 'one':
                formatted_data.append(self.format_one_shot(row, perturb_type=perturb_type))
            elif shot_type == 'few':
                formatted_data.append(self.format_few_shot(row, perturb_type=perturb_type))
            else:
                raise ValueError("Invalid shot type")

        return formatted_data

    def save_formatted_data_to_csv(self, formatted_data, output_filepath):
        """
        Save the formatted data to a CSV file.
        """
        df = pd.DataFrame(formatted_data)
        df.to_csv(output_filepath, index=False)
