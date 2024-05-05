from synthetic_data_generation.rag_data import DataGenerator

# Example usage
text = """Question: What is the capital of France?
Answer: Paris

Question: What is the tallest mountain in the world?
Answer: Mount Everest"""

qa_tuples = DataGenerator.extract_qa_tuples(text)

assert qa_tuples == [('What is the capital of France?', 'Paris'),
                     ('What is the tallest mountain in the world?', 'Mount Everest')]
