
#include <stdio.h>
#include <string.h>
#include <string>

#include "g2p.h"

#define MAX_WORD_LEN 35

using namespace std;

int main(int argc, char **argv) {

	G2P *g2p = new G2P(string(argv[1]), string(argv[2]), string(argv[3]));

	char word[MAX_WORD_LEN];
	while (fgets(word, MAX_WORD_LEN, stdin)) {
		char *trimmed_word = strtok(word, "\n");
		g2p->Phonetisize(string(trimmed_word));
	}

}
