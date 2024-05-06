epic:
self query retriever (SQR)

tickets:
implement answer scorer (done)
add md table of content (TOC) to chunk metadata (done)

route answer (irrelevant context)
- evaluate answer(question,answer) 
	-> yes | no
	- stop hallucinations

answer is bad
- evaluate documents(question,answer,doc) 
	-> relevant | not relevant
	- check toc() (done)
		-> relevant headers | document not relevant
			- get chunks from headers (done)

- reflection (done)

- route inadequate answer
	- reflection 
	- langchain agent with tools?
		- investigate
	- SQR?
    - reform query? how?

- need synthetic test data to see where model issues are (done)
	- will help to guide work on answer routing and SQR
	- use table of content?

	basic synthetic test data can be generated
	saved as csv of question,answer as well as txt file of question and answers

- generate more syn data covering more styles of md docs

- add ragas evaluation using syn test data (done for now)
	- wrap eval in an agent to create an evaluation object 
	- eval object should contain ragas metrics

	= ragas metrics may not be totally suitable 
	we are currently using eval and syn data to identify deficiencies in the rag chain
	(to possibly add more agents into the chain)
	ragas eval and metrics does not suit this purpose
		might be useful to compare llms and retrievers
		might be useful if syn data ground truth is more detailed
	
	more detail:
	- answer correctness metric measures TP,FP,FN but FP is usually due to the sparseness of the generated ground_truth answers

	other metrics

	generation metrics measure the performance of llm - our llm is powerful enough to not require measuring
	Faithfulness - factual consistency of the generated answer against the given context
	Answer relevancy - relevance of answer to question

	retrieval metrics measure the performance of retrieval - 
	Context recall - measures whether retrieved context aligns with the ground truth
	Context precision - evaluates whether all of the ground-truth relevant items present in the contexts are placed first

	Context relevancy - relevancy of the retrieved context
	Context entity recall - proportion of entities in ground truth that are in the context


### 6/5/2024
Trial2 indicates decent-good performance for the chatbot for longer form questions from questions about items in the documents.
Basic attempts at short-form questions show hit-or-miss performance for this style of questions.
Best next step is to probably generate synthetic short-form questions to identify deficiencies.

