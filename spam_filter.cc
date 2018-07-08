// spam filter using the Naive Bayes classifier API

#include <string>
#include <iostream>
#include <vector>
#include "./naive_bayes.h"

using std::string;
using std::cout;
using std::endl;
using std::vector;

#define SPAM_TRAINING_SET "spam_training_set.txt"
#define NON_SPAM_TRAINING_SET "non_spam_training_set.txt"

int main(int argc, char ** argv) {
    if (argc < 2) {
        cout << "usage: spam_filter <mail_file> [...]" << endl;
        return EXIT_FAILURE;
    }

    vector<string> files_to_classify;
    for (int i = 1; i < argc; i++) {
        files_to_classify.push_back(string(argv[i]));
    }

    int doc_count = files_to_classify.size();

    NaiveBayes classifier;
    classifier.train(SPAM_TRAINING_SET, std::string("SPAM"));
    classifier.train(NON_SPAM_TRAINING_SET, std::string("NON-SPAM"));

    for (int i = 0; i < doc_count; i++) {
	    std::string result = classifier.classify(files_to_classify.at(i));
	cout << result << " probability of \'" << files_to_classify.at(i) << "\' is: " <<
		classifier.getScore(result) << endl;
    }

    return EXIT_SUCCESS;
}
