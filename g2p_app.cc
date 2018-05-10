
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>

#include "g2p.h"

#define MAX_WORD_LEN 35

using namespace std;

int main(int argc, char **argv) {

	G2P *g2p = new G2P(string(argv[1]), string(argv[2]), string(argv[3]));

	char word[MAX_WORD_LEN];
	while (fgets(word, MAX_WORD_LEN, stdin)) {
		char *trimmed_word = strtok(word, "\n");
		vector<string> pron = g2p->Phonetisize(string(trimmed_word));
		fprintf(stderr, "%s\t", trimmed_word);
		for (vector<string>::iterator it = pron.begin(); it != pron.end(); it++) {
			if (it != pron.begin()) {
				fprintf(stderr, " ");
			}
			fprintf(stderr, "%s", (*it).c_str());
		}
		fprintf(stderr, "\n");
	}
}
