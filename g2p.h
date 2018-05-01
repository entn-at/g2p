
#ifndef INCLUDE_G2P_H_
#define INCLUDE_G2P_H_

#include <map>

#include "tensorflow/core/public/session.h"

class G2P {

public:
	G2P(const char *nn_path, const char *nn_meta);
	~G2P();

	void Phonetisize(const char *instr);

private:
	tensorflow::Session* _session;
	tensorflow::GraphDef _graph_def;
	char _nn_model_type[64];
	std::map<std::string, int> _g2i;
	std::map<int, std::string> _i2p;

	bool _best_nn_hyp = true;
	bool _nn_lattice = true;
};


#endif /* INCLUDE_G2P_H_ */
