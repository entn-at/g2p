
#include <string>
#include <vector>
#include <cstdint>
#include <cmath>
#include <map>
#include <unordered_map>

#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "include/PhonetisaurusScript.h"
#include "fst/fstlib.h"

#include "g2p.h"
#include "universal_lowercase.h"

#define MAX_WORD_LEN 33
#define BATCH_SIZE 256

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
	fclose(fp);
}

class G2P::Impl {
public:
	Impl(string nn_path, string nn_meta, string fst_path, string dict_path);
	~Impl();
	vector<vector<string> > Phonetisize(vector<string> words);
private:
	tensorflow::Session* _session = NULL;
	tensorflow::GraphDef _graph_def;
	char _nn_model_type[64];
	map<string, int> _g2i;
	map<int, string> _i2p;
	unordered_map<string, vector<string> > _dict;

	PhonetisaurusScript *_fst_decoder = NULL;

	bool IsStringOk(vector<string> &graphemes);
	vector<pair<string, Tensor> > PrepareNNInput(vector<vector<string> > &graphemes);
	void PhonetisizeNN(vector<vector<string> > &graphemes, vector<vector<string>* > &pron);
	void PhonetisizeFST(string word, vector<string> *pron);
	void PhonetisizeIntersect(vector<vector<string> > &graphemes, vector<string> &words,
			vector<vector<string>* > &pron);
	void GetShortestPath(VectorFst<StdArc> *res_fst, vector<string> *pron);
};

G2P::G2P(string nn_path, string nn_meta, string fst_path, string dict_path)
: pimpl(new Impl(nn_path, nn_meta, fst_path, dict_path)) {}

G2P::~G2P() {
	delete pimpl;
}

vector<vector<string> >
G2P::Phonetisize(vector<string> words) {
	return pimpl->Phonetisize(words);
}

static void
read_dict(string path, unordered_map<string, vector<string> > &dict) {
	FILE *fp = fopen(path.c_str(), "r");
	char line[1024];
	while (fgets(line, 1024, fp)) {
		char *word = strtok(line, "\t\n");
		char *phonemes = strtok(NULL, "\t\n");
		char *phoneme = strtok(phonemes, " \n");
		vector<string> pron;
		while (phoneme != NULL) {
			pron.push_back(string(phoneme));
			phoneme = strtok(NULL, " \n");
		}
		dict.insert(make_pair(string(word), pron));
	}
	fclose(fp);
}

G2P::Impl::Impl(string nn_path, string nn_meta, string fst_path, string dict_path) {
	if (!nn_path.empty()) {
		/* Load the nn model */
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
	}
	if (!fst_path.empty()) {
		/* Load fst model */
		_fst_decoder = new PhonetisaurusScript(fst_path, "");
	}
	if (!dict_path.empty()) {
		/* load dictionary */
		read_dict(dict_path, _dict);
	}
}

G2P::Impl::~Impl()
{
	if (_fst_decoder) {
		delete _fst_decoder;
	}
	if (_session) {
		_session->Close();
		delete _session;
	}
}

void
G2P::Impl::GetShortestPath(VectorFst<StdArc> *res_fst, vector<string> *pron) {
	StdVectorFst result;
	ShortestPath(*res_fst, &result);
	for (StateIterator<StdFst> siter(result); !siter.Done(); siter.Next()) {
		for (ArcIterator<StdFst> aiter(result, siter.Value()); !aiter.Done(); aiter.Next()) {
			int olabel = (int)aiter.Value().olabel;
			pron->push_back(_fst_decoder->FindOsym(olabel));
		}
	}
	reverse(pron->begin(), pron->end());
}

vector<pair<string, Tensor> >
G2P::Impl::PrepareNNInput(vector<vector<string> > &graphemes) {

	int max_len = 0;
	for (int i = 0; i < graphemes.size(); i++) {
		if (graphemes[i].size() > max_len) {
			max_len = graphemes[i].size();
		}
	}

	int stop_num = 0;
	if (strcmp(_nn_model_type, "ctc") == 0) {
		stop_num = 3;
	} else {
		stop_num = 1;
	}
	int max_len_extra = max_len + stop_num;

	Tensor input(DT_INT32, TensorShape({(int)graphemes.size(), max_len_extra}));
	Tensor slen_input(tensorflow::DT_INT32, tensorflow::TensorShape({(int)graphemes.size()}));
	auto input_map = input.tensor<int, 2>();
	auto slen_map = slen_input.tensor<int, 1>();
	for (int i = 0; i < graphemes.size(); i++) {
		slen_map(i) = graphemes[i].size() + stop_num;
		for (int j = 0; j < graphemes[i].size(); j++) {
			input_map(i, j) = _g2i.find(graphemes[i][j])->second;
		}
		for (int j = graphemes[i].size(); j < max_len_extra; j++) {
			input_map(i, j) = _g2i.size();
		}
	}

	vector<pair<string, Tensor> > inputs;
	inputs.push_back(std::make_pair("graphemes_ph", input));
	inputs.push_back(std::make_pair("grapeheme_seq_len_ph", slen_input));
	return inputs;
}

bool
G2P::Impl::IsStringOk(vector<string> &graphemes) {
	if (graphemes.empty()) {
		return false;
	}
	if (graphemes.size() >= MAX_WORD_LEN) {
		return false;
	}

	for (int i = 0; i < graphemes.size(); i++) {
		if (_session != NULL) {
			if (_g2i.find(graphemes[i]) == _g2i.end()) {
				return false;
			}
		} else if (_fst_decoder != NULL) {
			if (_fst_decoder->FindIsym(graphemes[i]) == -1) {
				return false;
			}
		}
	}
	return true;
}

static vector<string>
word_to_graphemes(string &instr) {
	vector<string> graphemes;
	for (size_t i = 0; i < instr.length();) {
		int cplen = 1;
		if((instr[i] & 0xf8) == 0xf0) cplen = 4;
		else if((instr[i] & 0xf0) == 0xe0) cplen = 3;
		else if((instr[i] & 0xe0) == 0xc0) cplen = 2;
		if((i + cplen) > instr.length()) cplen = 1;
		graphemes.push_back(instr.substr(i, cplen));
		i += cplen;
	}
	return graphemes;
}

vector<vector<string> >
G2P::Impl::Phonetisize(vector<string> words) {
	vector<vector<string> > pron(words.size());
	vector<vector<string>> graphemes_for_nn;
	vector<string> words_for_nn;
	vector<vector<string>* > pron_ptr;
	vector<vector<string> >::iterator pron_it = pron.begin();
	for (int i = 0; i < words.size(); i++, pron_it++) {
		/* bring word to lower case */
		string word = lowercase(words[i]);
		/* split the word into graphemes */
		vector<string> graphemes = word_to_graphemes(word);

		/* check if all the graphemes are supported */
		if (!IsStringOk(graphemes)) {
			continue;
		}

		/* if there is a dictionary, check if there is pronunciation there */
		if (_dict.size() > 0) {
			unordered_map<string, vector<string> >::const_iterator got = _dict.find(word);
			if (got != _dict.end()) {
				copy(got->second.begin(), got->second.end(), back_inserter(*pron_it));
				continue;
			}
		}

		if (_session == NULL) {
			/* Phonetisize word using fst */
			PhonetisizeFST(word, &(*pron_it));
		} else {

			/* Store word for phonetisation using NN in batch mode */
			graphemes_for_nn.push_back(graphemes);
			words_for_nn.push_back(word);
			pron_ptr.push_back(&(*pron_it));

			if (words_for_nn.size() >= BATCH_SIZE) {
				/* Perform phonetization of a batch  */
				if (_fst_decoder) {
					PhonetisizeIntersect(graphemes_for_nn, words_for_nn, pron_ptr);
				} else {
					PhonetisizeNN(graphemes_for_nn, pron_ptr);
				}
				graphemes_for_nn.clear();
				words_for_nn.clear();
				pron_ptr.clear();
			}
		}
	}

	if (_session && words_for_nn.size() > 0) {
		/* Perform phonetization of the left-over  */
		if (_fst_decoder) {
			PhonetisizeIntersect(graphemes_for_nn, words_for_nn, pron_ptr);
		} else {
			PhonetisizeNN(graphemes_for_nn, pron_ptr);
		}
	}

	return pron;
}

void
G2P::Impl::PhonetisizeNN(vector<vector<string> > &graphemes, vector<vector<string>* > &pron) {
	vector<pair<string, Tensor> > inputs = PrepareNNInput(graphemes);
	std::vector<tensorflow::Tensor> outputs;
	Status status = _session->Run(inputs, {"g2p/predicted_1best"}, {}, &outputs);
	if (!status.ok()) {
		fprintf(stderr, "**Error! Failed to run prediction\n");
		fprintf(stderr, "**Error! status: %s\n", status.ToString().c_str());
		exit(1);
	}
	auto phonemes_map = outputs[0].tensor<int, 2>();
	for (int i = 0; i < (int)outputs[0].dim_size(0); i++) {
		for (int j = 0; j < (int)outputs[0].dim_size(1); j++) {
			if (phonemes_map(i, j) >= _i2p.size()) {
				if (strcmp(_nn_model_type, "ctc") == 0) {
					continue;
				} else if (strcmp(_nn_model_type, "attention") == 0 ||
						strcmp(_nn_model_type, "transformer") == 0) {
					break;
				} else {
					fprintf(stderr, "**Error! Unexpected model type %s\n", _nn_model_type);
				}
			}
			string phone = _i2p.find((int)phonemes_map(i, j))->second;
			pron[i]->push_back(phone);
		}
	}
}

void
G2P::Impl::PhonetisizeFST(string word, vector<string> *pron) {
	vector<PathData> pathData = _fst_decoder->Phoneticize(word);
	for (int i = 0; i < pathData[0].Uniques.size(); i++) {
		pron->push_back(_fst_decoder->FindOsym(pathData[0].Uniques[i]));
	}
}

void
G2P::Impl::PhonetisizeIntersect(vector<vector<string> > &graphemes,
		vector<string> &words, vector<vector<string>* > &pron) {
	vector<pair<string, Tensor> > inputs = PrepareNNInput(graphemes);
	std::vector<tensorflow::Tensor> outputs;
	Status status = _session->Run(inputs, {"g2p/probs"}, {}, &outputs);
	if (!status.ok()) {
		fprintf(stderr, "**Error! Failed to run prediction\n");
		fprintf(stderr, "**Error! status: %s\n", status.ToString().c_str());
		exit(1);
	}
	auto probs_map = outputs[0].tensor<float, 3>();
	for (int i = 0; i < (int)outputs[0].dim_size(0); i++) {
		/* Create fst that will be filled by nn outputs */
		int new_state = 1;
		StdVectorFst nn_fst;
		nn_fst.AddState();
		nn_fst.SetStart(0);

		/* Convert predictions of NN to FST lattice */
		for (int j = 0; j < (int)outputs[0].dim_size(1); j++) {
			int layer_start = new_state;
			for (int k = 0; k < (int)outputs[0].dim_size(2); k++) {
				string olabel = "";
				if (k >= _i2p.size()) {
					olabel = "<eps>";
				} else {
					olabel =  _i2p.find(k)->second;
				}
				int olabeli = _fst_decoder->FindOsym(olabel);
				if (j == 0) {
					nn_fst.AddArc(0, StdArc(olabeli, olabeli, -log(probs_map(i, j, k)), new_state));
					nn_fst.AddState();
					new_state++;
				} else {
					int diff = (int)outputs[0].dim_size(2);
					for (int l = 0; l < (int)outputs[0].dim_size(2); l++) {
						nn_fst.AddArc(layer_start - diff + l, StdArc(olabeli, olabeli, -log(probs_map(i, j, k)), new_state));
					}
					nn_fst.AddState();
					new_state++;
				}
			}

			if (strcmp(_nn_model_type, "ctc") != 0 && j > 0) {
				/* for attention and transformer - break when stop symbol is emmited */
				float max_prob = 0.0;
				int most_probable = -1;
				for (int k = 0; k < (int)outputs[0].dim_size(2); k++) {
					if (probs_map(i, j, k) > max_prob) {
						max_prob = probs_map(i, j, k);
						most_probable = k;
					}
				}
				if (most_probable >= _i2p.size()) {
					break;
				}
			}
		}

		/* Add final state to NN FST lattice */
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

		/* Phonetisize using FST model */
		VectorFst<StdArc>* fst = _fst_decoder->GetLattice(words[i]);
		bool start_state = true;
		int empty_label = _fst_decoder->FindOsym(string("_"));
		int last_state = -1;
		int extra_state = fst->NumStates();
		/* Copy lattice, expanding phoneme compounds */
		StdVectorFst fst_cpy;
		for (int j = 0; j < extra_state; j++) {
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
		RmEpsilon(&fst_cpy);
		Project(&fst_cpy, ProjectType::PROJECT_OUTPUT);
		ArcSort(&fst_cpy, OLabelCompare<StdArc>());

		/* Finally perform intersection */
		VectorFst<StdArc>* res_fst = new VectorFst<StdArc>();
		Intersect(nn_fst, fst_cpy, res_fst);
		RmEpsilon(res_fst);

		GetShortestPath(res_fst, pron[i]);
		/* clean created tmp fsts */
		delete fst;
		delete res_fst;
	}
}
