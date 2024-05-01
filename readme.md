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
	- check toc()
		-> relevant headers | document not relevant
			- get chunks from headers
			
	- reform query? how?
	
- 