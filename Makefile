SPAM_FILTER=spam_filter.cc
CLASSIFIER=naive_bayes.h

CXX=g++
CXX_FLAGS=-g --std=c++17 -Wall

goal: spam_filter

spam_filter: ${SPAM_FILTER} ${CLASSIFIER}
	${CXX} ${CXX_FLAGS} -o spam_filter ${SPAM_FILTER} ${CLASSIFIER}

clean:
	rm -f spam_filter
