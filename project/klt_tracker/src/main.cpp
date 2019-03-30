#include <Eigen/Dense>
#include <iostream>
#include "hello/hello.hpp"

int main() {

	hello::say_hello();
	Eigen::MatrixXd d;

	Eigen::Matrix3d f;

	f = Eigen::Matrix3d::Random();

	std::cout << f << std::endl;

}
