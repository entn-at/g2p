
#ifndef INCLUDE_G2P_H_
#define INCLUDE_G2P_H_

#include <string>
#include <vector>

class G2P {

public:
	G2P(std::string nn_path, std::string nn_meta, std::string fst_path,
			std::string dict_path);
	~G2P();

	std::vector<std::vector<std::string> > Phonetisize(std::vector<std::string> words);
	std::vector<std::string> GetGraphemes();
	std::vector<std::string> GetPhonemes();

private:
	class Impl;
	Impl *pimpl;
};


#endif /* INCLUDE_G2P_H_ */
