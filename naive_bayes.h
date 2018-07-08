/**
 * Naive Bayes classifier API
 * @author: parth_shel
 * @version: v:1.1 - July 7, 2018
 **/

#ifndef NAIVE_BAYES_H
#define NAIVE_BAYES_H

#include <math.h>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <utility>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>

#define DEBUG_MODE true

#define STOP_LENGTH 1
#define STOP_WORDS "stopwords.txt"

class NaiveBayes {
 public:
        NaiveBayes();
        void train(std::string file_path, std::string label);
	std::string classify(std::string file_path);
	double getScore(std::string label);

 private:
        bool isTrained;
        std::set<std::string> labels;
        std::set<std::string> stopWords;
        std::vector<char> punctuations = {
            '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+',
            ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@',
            '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~'};
        std::map<std::pair<std::string, std::string>, unsigned int> stemLabelCounts;
        std::map<std::string, unsigned int> docCounts;
        std::map<std::string, unsigned int> stemCounts;
        std::map<std::string, double> scores;

        void getStopWords(std::string set);
        std::vector<std::string> tokenize(std::string text);
        std::string extractWinner();
        std::string guess(std::string text);
        unsigned int stemLabelCount(std::string stem, std::string label);
        unsigned int stemInverseLabelCount(std::string stem, std::string label);
        unsigned int stemTotalCount(std::string stem);
        unsigned int docCount(std::string label);
        unsigned int docInverseCount(std::string label);
        void incrementStemCount(std::string stem, std::string label);
        void incrementDocCount(std::string label);
        std::string readText(std::string file_path);
};

NaiveBayes::NaiveBayes() {
    isTrained = false;
    getStopWords(std::string(STOP_WORDS));
}

void NaiveBayes::train(std::string file_path, std::string label) {
    labels.insert(label);  // register label

    std::vector<std::string> words = tokenize(readText(file_path));  // bag of words

    std::vector<std::string>::iterator itr;
    for (itr = words.begin(); itr != words.end(); ++itr) {
        incrementStemCount(*itr, label);
    }
    incrementDocCount(label);

    isTrained = true;
    return;
}

std::string NaiveBayes::classify(std::string file_path) {
    if (!isTrained) {
        return (std::string) NULL;
    }

    scores.clear();
    std::string result = guess(readText(file_path));

    if (DEBUG_MODE) {
        std::cout << "\"" << file_path << "\" is classified as " << result
            << " with a confidence level of " << getScore(result) * 100 << "%" << std::endl;
    }

    return  result;
}

void NaiveBayes::getStopWords(std::string set) {
    stopWords.clear();
    std::ifstream file(set, std::ios::in);
    std::string word;
    while (std::getline(file, word)) {
        stopWords.insert(word);
    }
    file.close();
    return;
}

std::vector<std::string> NaiveBayes::tokenize(std::string text) {
    // remove punctuations and unwanted characters
    for (unsigned int i = 0; i < punctuations.size(); i++) {
        text.erase(remove(text.begin(), text.end(), punctuations.at(i)), text.end());
    }
    text.erase(std::remove_if(text.begin(), text.end(), &isdigit), text.end());

    // convert to lowercase, remove extra whitespaces and finally split using the spaces
    std::transform(text.begin(), text.end(), text.begin(), ::tolower);

    std::replace(text.begin(), text.end(), '\n', ' ');
    std::replace(text.begin(), text.end(), '\r', ' ');
    std::replace(text.begin(), text.end(), '\b', ' ');
    std::replace(text.begin(), text.end(), '\t', ' ');
    std::replace(text.begin(), text.end(), '\v', ' ');

    std::string cleansed_text;
    std::unique_copy(text.begin(), text.end(),
            std::back_insert_iterator<std::string> (cleansed_text),
            [](char a, char b){ return isspace(a) && isspace(b);});

    std::stringstream ss(cleansed_text);
    std::string token;
    std::vector<std::string> items;
    while (std::getline(ss, token, ' ')) {
        items.push_back(token);
    }

    // remove duplicate words
    items.erase(unique(items.begin(), items.end()), items.end());

    std::vector<std::string> tokens;

    // remove stop words
    std::vector<std::string>::iterator itr;
    for (itr = items.begin(); itr != items.end(); ++itr) {
        std::string toCheck = *itr;
        if (stopWords.count(toCheck) || toCheck.length() <= STOP_LENGTH) {
            continue;
        }
        tokens.push_back(toCheck);
    }

    return tokens;
}

std::string NaiveBayes::extractWinner() {
    double bestScore = 0;
    std::string bestLabel;

    std::map<std::string, double>::iterator itr;
    for (itr = scores.begin(); itr != scores.end(); ++itr) {
        if (itr->second > bestScore) {
            bestScore = itr->second;
            bestLabel = itr->first;
        }
    }

    return bestLabel;
}

double NaiveBayes::getScore(std::string label) {
    double score = 0;
    if (scores.count(label)) {
        score = scores[label];
    }
    return score;
}

std::string NaiveBayes::guess(std::string text) {
    std::vector<std::string> words = tokenize(text);
    unsigned int totalDocCount = 0;
    std::map<std::string, unsigned int> _docCounts;
    std::map<std::string, unsigned int> _docInverseCounts;
    std::map<std::string, unsigned int> _labelProbability;

    std::set<std::string>::iterator itr;
    for (itr = labels.begin(); itr != labels.end(); ++itr) {
        std::string label = *itr;
        _docCounts[label] = docCount(label);
        _docInverseCounts[label] = docInverseCount(label);
        totalDocCount += _docCounts[label];
    }

    std::set<std::string>::iterator labels_itr;
    for (labels_itr = labels.begin(); labels_itr != labels.end(); ++labels_itr) {
        std::string label = *labels_itr;
        double logSum = 0;
        _labelProbability[label] = _docCounts[label] / totalDocCount;

        std::vector<std::string>::iterator words_itr;
        for (words_itr = words.begin(); words_itr != words.end(); ++words_itr) {
            std::string word = *words_itr;
            unsigned int _stemTotalCount = stemTotalCount(word);
            if (_stemTotalCount == 0) {
                continue;
            } else {
                // MAGIC! don't touch
                double wordProbability =
                    stemLabelCount(word, label) / _docCounts[label];
                double wordInverseProbability =
                    stemInverseLabelCount(word, label) / _docInverseCounts[label];
                double wordicity =
                    wordProbability / (wordProbability + wordInverseProbability);

                wordicity = (0.5 + (_stemTotalCount * wordicity)) / (1 + _stemTotalCount);
                if (wordicity == 0) {
                    wordicity = 0.01;
                } else if (wordicity == 1) {
                    wordicity = 0.99;
                }
                logSum += log(1 - wordicity) - log(wordicity);

                if (DEBUG_MODE) {
                    std::cout << label << " -\'icity/\'ivity of \"" << word
                        << "\" is " << wordicity << std::endl;
                }
            }
        }  // words_itr
        scores[label] = 1 / (1 + exp(logSum));
    }  // labels_itr

    return extractWinner();
}

unsigned int NaiveBayes::stemLabelCount(std::string stem, std::string label) {
    unsigned int count = 0;
    if (stemLabelCounts.count(std::make_pair(stem, label))) {
        count = stemLabelCounts[std::make_pair(stem, label)];
    }
    return count;
}

unsigned int NaiveBayes::stemInverseLabelCount(std::string stem, std::string label) {
    unsigned int total = 0;
    std::set<std::string>::iterator itr;
    for (itr = labels.begin(); itr != labels.end(); ++itr) {
        if (label.compare(*itr) == 0) {
            continue;
        }
        total += stemLabelCount(stem, *itr);
    }
    return total;
}

unsigned int NaiveBayes::stemTotalCount(std::string stem) {
    unsigned int count = 0;
    if (stemCounts.count(stem)) {
        count = stemCounts[stem];
    }
    return count;
}

unsigned int NaiveBayes::docCount(std::string label) {
    unsigned int count = 0;
    if (docCounts.count(label)) {
        count = docCounts[label];
    }
    return count;
}

unsigned int NaiveBayes::docInverseCount(std::string label) {
    unsigned int total = 0;
    std::set<std::string>::iterator itr;
    for (itr = labels.begin(); itr != labels.end(); ++itr) {
        if (label.compare(*itr) == 0) {
            continue;
        }
        total += docCount(*itr);
    }
    return total;
}

void NaiveBayes::incrementStemCount(std::string stem, std::string label) {
    stemLabelCounts[std::make_pair(stem, label)]++;
    stemCounts[stem]++;
}

void NaiveBayes::incrementDocCount(std::string label) {
    docCounts[label]++;
}

std::string NaiveBayes::readText(std::string file_path) {
    std::ifstream file(file_path, std::ios::in);
    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    return text;
}

#endif  // NAIVE_BAYES_H
