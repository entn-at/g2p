
#include <string>
#include <vector>
#include <cstdint>

#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

#include "g2p.h"

using namespace std;
using namespace tensorflow;

static void
read_nn_meta(const char *path, char *model_type, std::map<string, int> *g2i,
		std::map<int, string> *i2p)
{
	char line[1024];
	FILE *fp = fopen(path, "r");

	if (!fgets(line, 1024, fp)) {
		fprintf(stderr, "**Error! Failed to read model type\n");
	}
	strcpy(model_type, strtok(line, "\n"));

	/* read graphemes mapping */
	if (!fgets(line, 1024, fp)) {
		fprintf(stderr, "**Error! Failed to read graphemes\n");
	}
	int idx = 0;
	char *grapheme = strtok(line, " \n");
	while (grapheme != NULL) {
		g2i->insert(pair<string, int>(string(grapheme), idx++));
		grapheme = strtok(NULL, " \n");
	}

	/* read phoneme mapping */
	if (!fgets(line, 1024, fp)) {
		fprintf(stderr, "**Error! Failed to read phonemes\n");
	}
	idx = 0;
	char *phoneme = strtok(line, " \n");
	while (phoneme != NULL) {
		i2p->insert(pair<int, string>(idx++, string(phoneme)));
		phoneme = strtok(NULL, " \n");
	}
}

G2P::G2P(const char *nn_path, const char *nn_meta)
{
	/* Load the model */
	Status status = NewSession(SessionOptions(), &_session);
	if (!status.ok()) {
		fprintf(stderr, "**Error! Failed to create new session\n");
		exit(1);
	}
	status = ReadBinaryProto(tensorflow::Env::Default(), string(nn_path), &_graph_def);
	if (!status.ok()) {
		fprintf(stderr, "**Error! Failed to read graph\n");
		exit(1);
	}
	status = _session->Create(_graph_def);
	if (!status.ok()) {
		fprintf(stderr, "**Error! Failed to add graph to the session\n");
		fprintf(stderr, "**Error! status: %s\n", status.ToString().c_str());
		exit(1);
	}

	read_nn_meta(nn_meta, _nn_model_type, &_g2i, &_i2p);
}

G2P::~G2P()
{
}

void
G2P::Phonetisize(const char *instr)
{
	int instr_len = strlen(instr);
	if (instr_len == 0) {
		return;
	}
	fprintf(stderr, "%s", instr);
	vector<pair<string, Tensor> > inputs;
	int alloc_len = instr_len;
	if (strcmp(_nn_model_type, "ctc") == 0) {
		alloc_len += 3;
	}
	Tensor input(DT_INT32, TensorShape({1, alloc_len}));
	auto input_map = input.tensor<int, 2>();

	char grapheme[2];
	grapheme[1] = '\0';
	for (int i = 0; i < instr_len; i++) {
		grapheme[0] = instr[i];
		input_map(0, i) = _g2i.find(string(grapheme))->second;
	}
	if (strcmp(_nn_model_type, "ctc") == 0) {
		input_map(0, instr_len) = _g2i.size();
		input_map(0, instr_len + 1) = _g2i.size();
		input_map(0, instr_len + 2) = _g2i.size();
	}
	inputs.push_back(std::make_pair("graphemes_ph", input));
	tensorflow::Tensor slen_input(tensorflow::DT_INT32, tensorflow::TensorShape({1}));
	auto slen_map = slen_input.tensor<int, 1>();
	slen_map(0) = alloc_len;
	inputs.push_back(std::make_pair("grapeheme_seq_len_ph", slen_input));

	std::vector<tensorflow::Tensor> outputs;
	Status status = _session->Run(inputs, {"g2p/predicted_1best"}, {}, &outputs);

	/*auto pmap = outputs[0].tensor<float,3>();
	for (int i = 0; i < (int)outputs[0].dim_size(1); i++) {
		for (int j = 0; j < (int)outputs[0].dim_size(2); j++) {
			fprintf(stderr, "%f ", pmap(0, i, j));
		}
		fprintf(stderr, "\n");
	}*/

	if (!status.ok()) {
		fprintf(stderr, "**Error! Failed to run prediction\n");
		fprintf(stderr, "**Error! status: %s\n", status.ToString().c_str());
		exit(1);
	}
	auto phonemes_map = outputs[0].tensor<int, 2>();
	bool first_space = true;
	for (int i = 0; i < (int)outputs[0].dim_size(1); i++) {
		if (phonemes_map(0, i) >= _i2p.size()) {
			if (strcmp(_nn_model_type, "ctc") == 0) {
				continue;
			} else if (strcmp(_nn_model_type, "attention") == 0 ||
					strcmp(_nn_model_type, "transformer") == 0) {
				break;
			} else {
				fprintf(stderr, "**Error! Unexpected model type %s\n", _nn_model_type);
			}
		}
		fprintf(stderr, "%s", (first_space ? "\t" : " "));
		first_space = false;
		fprintf(stderr, "%s", _i2p.find((int)phonemes_map(0, i))->second.c_str());
	}
	fprintf(stderr, "\n");
}
