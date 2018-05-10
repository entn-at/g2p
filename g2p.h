
#ifndef INCLUDE_G2P_H_
#define INCLUDE_G2P_H_

#include <string>
#include <vector>

class G2P {

public:
	G2P(std::string nn_path, std::string nn_meta, std::string fst_path);
	~G2P();

	std::vector<std::string> Phonetisize(std::string instr);

private:
	class Impl;
	Impl *pimpl;
};


#endif /* INCLUDE_G2P_H_ */
