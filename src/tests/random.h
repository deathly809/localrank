
#include <cstdlib>
#include <cmath>
#include <limits>

#include <vector>


template<typename T>
T generateGaussianNoise(T mu, T sigma)
{
	const T epsilon = std::numeric_limits<T>::min();
	const T two_pi = 2.0*3.14159265358979323846;

	static T z0, z1;
	static bool generate;
	generate = !generate;

	if (!generate)
	   return z1 * sigma + mu;

	T u1, u2;
	do
	 {
	   u1 = rand() * (1.0 / RAND_MAX);
	   u2 = rand() * (1.0 / RAND_MAX);
	 }
	while ( u1 <= epsilon );

	z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
	z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
	return z0 * sigma + mu;
}

template <typename T>
std::vector<T> generateGaussianVectorNoise(size_t ElementWidth,const std::vector<T> & means, const std::vector<T> & sigmas) {
	std::vector<T> result;
    for(int elementPos = 0 ; elementPos < ElementWidth; ++elementPos) {
		result.push_back(generateGaussianNoise(means[elementPos],sigmas[elementPos]));
    }
    return result;
}