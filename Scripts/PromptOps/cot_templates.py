
import pandas as pd
import pickle
from PromptOps.perturb import Perturbation

class COTTemplateFormatter:
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
        if perturb_type == 'robust':
            return question  # For robust, specific perturbations are expected from the data
        elif perturb_type == 'taxonomy':
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
            # original = f"Question: {row['Original_Question']}\nElaborated Sentence: {row['Elaborated_sentence']}"
            # perturbed = f"Question: {row['Perturbed_Question']}\nElaborated Sentence: {row['Elaborated_sentence']}"
            
            original = f"{row['Prefix']}\nContext: {row['Context']}\n{row['Question']}"
            perturbed = f"{row['Prefix']}\nContext: {row['Context']}\n{question}"
            
            
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
            elaborated_sentence = row['Elaborated_sentence']
            if perturb_type:
                question = self.perturb_question(question, perturb_type)
            original = f"Question: {row['Question']}\nElaborated Sentence: {row['Elaborated_sentence']}"
            perturbed = f"Question: {question}\nElaborated Sentence: {row['Elaborated_sentence']}"
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
            original = f"Question 1: {row['Question_1']}\nElaborated Sentence 1: {row['Elaborated_sentence_1']}\nAnswer 1: {row['Answer_1']}\n\nQuestion: {row['Original_Question']}\nElaborated Sentence: {row['Elaborated_sentence']}"
            perturbed = f"Question 1: {row['Question_1']}\nElaborated Sentence 1: {row['Elaborated_sentence_1']}\nAnswer 1: {row['Answer_1']}\n\nQuestion: {row['Perturbed_Question']}\nElaborated Sentence: {row['Elaborated_sentence']}"
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
            elaborated_sentence = row['Elaborated_sentence']
            if perturb_type:
                question = self.perturb_question(question, perturb_type)
            original = f"Question 1: {row['Question_1']}\nElaborated Sentence 1: {row['Elaborated_sentence_1']}\nAnswer 1: {row['Answer_1']}\n\nQuestion: {row['Question']}\nElaborated Sentence: {row['Elaborated_sentence']}"
            perturbed = f"Question 1: {row['Question_1']}\nElaborated Sentence 1: {row['Elaborated_sentence_1']}\nAnswer 1: {row['Answer_1']}\n\nQuestion: {question}\nElaborated Sentence: {row['Elaborated_sentence']}"
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
            few_shot_context = f"Question 1: {row['Question_1']}\nElaborated Sentence 1: {row['Elaborated_sentence_1']}\nAnswer 1: {row['Answer_1']}\n"
            if 'Question_2' in row and pd.notna(row['Question_2']):
                few_shot_context += f"Question 2: {row['Question_2']}\nElaborated Sentence 2: {row['Elaborated_sentence_2']}\nAnswer 2: {row['Answer_2']}\n"

            original = f"{few_shot_context}Question: {row['Original_Question']}\nElaborated Sentence: {row['Elaborated_sentence']}"
            perturbed = f"{few_shot_context}Question: {row['Perturbed_Question']}\nElaborated Sentence: {row['Elaborated_sentence']}"
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
            elaborated_sentence = row['Elaborated_sentence']
            if perturb_type:
                question = self.perturb_question(question, perturb_type)

            few_shot_context = f"Question 1: {row['Question_1']}\nElaborated Sentence 1: {row['Elaborated_sentence_1']}\nAnswer 1: {row['Answer_1']}\n"
            if 'Question_2' in row and pd.notna(row['Question_2']):
                few_shot_context += f"Question 2: {row['Question_2']}\nElaborated Sentence 2: {row['Elaborated_sentence_2']}\nAnswer 2: {row['Answer_2']}\n"

            original = f"{few_shot_context}Question: {row['Question']}\nElaborated Sentence: {row['Elaborated_sentence']}"
            perturbed = f"{few_shot_context}Question: {question}\nElaborated Sentence: {row['Elaborated_sentence']}"
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
                formatted_row = self.format_zero_shot(row, perturb_type=perturb_type)
            elif shot_type == 'one':
                formatted_row = self.format_one_shot(row, perturb_type=perturb_type)
            elif shot_type == 'few':
                formatted_row = self.format_few_shot(row, perturb_type=perturb_type)
            else:
                raise ValueError("Invalid shot type")
            formatted_data.append(formatted_row)

        return formatted_data

    def save_formatted_data_to_csv(self, formatted_data, output_filepath):
        """
        Save the formatted data to a CSV file.
        """
        df = pd.DataFrame(formatted_data)
        df.to_csv(output_filepath, index=False)
