
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>

#include "g2p.h"

#define MAX_WORD_LEN 35

using namespace std;

int main(int argc, char **argv) {

	G2P *g2p = new G2P(string(argv[1]), string(argv[2]), string(argv[3]),
			string(argv[4]));

	vector<string> words;
	char word[MAX_WORD_LEN];
	while (fgets(word, MAX_WORD_LEN, stdin)) {
		char *trimmed_word = strtok(word, "\n");
		words.push_back(string(trimmed_word));
	}

	vector<vector<string> > pron = g2p->Phonetisize(words);
	for (int i = 0; i < words.size(); i++) {
		for (vector<string>::iterator it = pron[i].begin(); it != pron[i].end(); it++) {
			if (it != pron[i].begin()) {
				fprintf(stderr, " ");
			}
			fprintf(stderr, "%s", (*it).c_str());
		}
		fprintf(stderr, "\n");
	}
}
