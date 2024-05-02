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

- need synthetic test data to see where model issues are
	- will help to guide work on answer routing and SQR