
#include <string>
#include <vector>
#include <cstdint>
#include <cmath>
#include <map>

#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "include/PhonetisaurusScript.h"
#include "fst/fstlib.h"

#include "g2p.h"

using namespace std;
using namespace fst;
using namespace tensorflow;

static void
read_nn_meta(string path, char *model_type, std::map<string, int> *g2i,
		std::map<int, string> *i2p)
{
	char line[1024];
	FILE *fp = fopen(path.c_str(), "r");

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

class G2P::Impl {
public:
	Impl(string nn_path, string nn_meta, string fst_path);
	~Impl();
	void Phonetisize(string instr);
private:
	tensorflow::Session* _session;
	tensorflow::GraphDef _graph_def;
	char _nn_model_type[64];
	std::map<std::string, int> _g2i;
	std::map<int, std::string> _i2p;

	PhonetisaurusScript *_fst_decoder;

	bool _best_nn_hyp = false;
	bool _nn_lattice = true;

	void PrintShortestPath(VectorFst<StdArc> *res_fst, std::string name);
};

G2P::G2P(string nn_path, string nn_meta, string fst_path)
: pimpl(new Impl(nn_path, nn_meta, fst_path)) {}

G2P::~G2P() {
	delete pimpl;
}

void G2P::Phonetisize(string instr) {
	pimpl->Phonetisize(instr);
}

G2P::Impl::Impl(string nn_path, string nn_meta, string fst_path) {
	/* Load the model */
	Status status = NewSession(SessionOptions(), &_session);
	if (!status.ok()) {
		fprintf(stderr, "**Error! Failed to create new session\n");
		exit(1);
	}
	status = ReadBinaryProto(tensorflow::Env::Default(), nn_path, &_graph_def);
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

	_fst_decoder = new PhonetisaurusScript(fst_path, "");
}

G2P::Impl::~Impl()
{
}

void G2P::Impl::PrintShortestPath(VectorFst<StdArc> *res_fst, string name) {
	StdVectorFst result;
	ShortestPath(*res_fst, &result);
	vector<string> pron;
	int prev_label = -999;
	for (StateIterator<StdFst> siter(result); !siter.Done(); siter.Next()) {
		for (ArcIterator<StdFst> aiter(result, siter.Value()); !aiter.Done(); aiter.Next()) {
			int olabel = (int)aiter.Value().olabel;
			if (prev_label == olabel) {
				continue;
			}
			prev_label = olabel;
			pron.push_back(_fst_decoder->FindOsym(olabel));
		}
	}
	reverse(pron.begin(), pron.end());
	for (vector<string>::iterator it = pron.begin(); it != pron.end(); it++) {
		if (it != pron.begin()) {
			fprintf(stderr, " ");
		}
		fprintf(stderr, "%s", (*it).c_str());
	}
	fprintf(stderr, "\n");
}

void
G2P::Impl::Phonetisize(string instr)
{
	int instr_len = instr.length();
	if (instr_len == 0) {
		return;
	}
	fprintf(stderr, "%s\t", instr.c_str());
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

	if (_best_nn_hyp) {
		std::vector<tensorflow::Tensor> outputs;
		Status status = _session->Run(inputs, {"g2p/predicted_1best"}, {}, &outputs);
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

	if (_nn_lattice) {
		std::vector<tensorflow::Tensor> outputs;
		Status status = _session->Run(inputs, {"g2p/probs"}, {}, &outputs);
		if (!status.ok()) {
			fprintf(stderr, "**Error! Failed to run prediction\n");
			fprintf(stderr, "**Error! status: %s\n", status.ToString().c_str());
			exit(1);
		}
		auto probs_map = outputs[0].tensor<float, 3>();
		int new_state = 1;
		StdVectorFst nn_fst;
		nn_fst.AddState();
		nn_fst.SetStart(0);
		int prev_most_probable = -1;
		for (int i = 0; i < (int)outputs[0].dim_size(1); i++) {
			/* Collapse same phonemes in raw output */
			float max_prob = 0.0;
			int most_probable = -1;
			for (int j = 0; j < (int)outputs[0].dim_size(2); j++) {
				if (probs_map(0, i, j) > max_prob) {
					max_prob = probs_map(0, i, j);
					most_probable = j;
				}
			}
			if (prev_most_probable == most_probable) {
				continue;
			}
			prev_most_probable = most_probable;

			int layer_start = new_state;
			for (int j = 0; j < (int)outputs[0].dim_size(2); j++) {
				string olabel = "";
				if (j >= _i2p.size()) {
					olabel = "<eps>";
				} else {
					olabel =  _i2p.find(j)->second;
				}
				int olabeli = _fst_decoder->FindOsym(olabel);
				if (i == 0) {
					nn_fst.AddArc(0, StdArc(olabeli, olabeli, -log(probs_map(0, i, j)), new_state));
					nn_fst.AddState();
					new_state++;
				} else {
					int diff = (int)outputs[0].dim_size(2);
					for (int k = 0; k < (int)outputs[0].dim_size(2); k++) {
						nn_fst.AddArc(layer_start - diff + k, StdArc(olabeli, olabeli, -log(probs_map(0, i, j)), new_state));
					}
					nn_fst.AddState();
					new_state++;
				}
			}

			if (strcmp(_nn_model_type, "ctc") != 0) {
				/* for attention and transformer - break when stop symbol is emmited */
				float max_prob = 0.0;
				int most_probable = -1;
				for (int j = 0; j < (int)outputs[0].dim_size(2); j++) {
					if (probs_map(0, i, j) > max_prob) {
						max_prob = probs_map(0, i, j);
						most_probable = j;
					}
				}
				if (most_probable >= (int)outputs[0].dim_size(2)) {
					break;
				}
			}

		}
		int diff = (int)outputs[0].dim_size(2);
		int olabeli = _fst_decoder->FindOsym(string("<eps>"));
		for (int k = 0; k < (int)outputs[0].dim_size(2); k++) {
			nn_fst.AddArc(new_state - diff + k, StdArc(olabeli, olabeli, 0.0, new_state));
		}
		nn_fst.AddState();
		nn_fst.SetFinal(new_state, 0.0);
		nn_fst.SetInputSymbols(_fst_decoder->osyms_);
		nn_fst.SetOutputSymbols(_fst_decoder->osyms_);
		RmEpsilon(&nn_fst);
		ArcSort(&nn_fst, OLabelCompare<StdArc>());

		VectorFst<StdArc>* fst = _fst_decoder->GetLattice(instr);
		StdVectorFst fst_cpy;
		bool start_state = true;
		int empty_label = _fst_decoder->FindOsym(string("_"));
		int last_state = -1;
		int extra_state = fst->NumStates();
		for (int i = 0; i < extra_state; i++) {
			fst_cpy.AddState();
		}
		for (StateIterator<StdFst> siter(*fst); !siter.Done(); siter.Next()) {
			if (start_state) {
				fst_cpy.SetStart(siter.Value());
				start_state = false;
			}
			last_state = (int)siter.Value();
			for (ArcIterator<StdFst> aiter(*fst, siter.Value()); !aiter.Done(); aiter.Next()) {
				const StdArc &arc = aiter.Value();
				int label = ((int)arc.olabel == empty_label) ? 0 : arc.olabel;
				/* Check if need to expand to monophones */
				string label_str = _fst_decoder->FindOsym(label);
				int sep_pos = label_str.find("|");
				if (sep_pos != string::npos) {
					string phone1 = label_str.substr(0, sep_pos);
					string phone2 = label_str.substr(sep_pos + 1);
					int phone1_i = _fst_decoder->FindOsym(phone1);
					int phone2_i = _fst_decoder->FindOsym(phone2);
					fst_cpy.AddArc(siter.Value(), StdArc(phone1_i, phone1_i, arc.weight, extra_state));
					fst_cpy.AddState();
					fst_cpy.AddArc(extra_state, StdArc(phone2_i, phone2_i, 0.0, arc.nextstate));
					extra_state++;
				} else {
					fst_cpy.AddArc(siter.Value(), StdArc(label, label, arc.weight, arc.nextstate));
				}
			}
			fst_cpy.SetFinal(last_state, fst->Final(last_state));
		}
		fst_cpy.SetInputSymbols(_fst_decoder->osyms_);
		fst_cpy.SetOutputSymbols(_fst_decoder->osyms_);

		RmEpsilon(fst);
		Project(fst, ProjectType::PROJECT_OUTPUT);
		ArcSort(fst, OLabelCompare<StdArc>());

		RmEpsilon(&fst_cpy);
		Project(&fst_cpy, ProjectType::PROJECT_OUTPUT);
		ArcSort(&fst_cpy, OLabelCompare<StdArc>());

		VectorFst<StdArc>* res_fst = new VectorFst<StdArc>();
		Intersect(nn_fst, fst_cpy, res_fst);
		RmEpsilon(res_fst);

		PrintShortestPath(res_fst, string("res"));

		delete fst;
		delete res_fst;
	}
}
