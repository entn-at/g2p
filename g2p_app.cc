
#include <stdio.h>

#include "g2p.h"

#define MAX_WORD_LEN 35


int main(int argc, char **argv) {

	G2P *g2p = new G2P(argv[1], argv[2]);

	char word[MAX_WORD_LEN];
	while (fgets(word, MAX_WORD_LEN, stdin)) {
		char *trimmed_word = strtok(word, "\n");
		g2p->Phonetisize(trimmed_word);
	}

}
