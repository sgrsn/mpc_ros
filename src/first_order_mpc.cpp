#include <iostream>
#include <Eigen/Core>
#include <qpOASES.hpp>

int main()
{
  // Define the first-order system
  // m v_dot + c v = u
  // m: mass
  // c: friction
  // u: input force
  // v: velosity
  double m = 1.0; // mass [kg]
  double c = 0.1; // friction [N/(m/s)]
  double v0 = 1.0;  // initial speed[m/s]

  // Set System
  // x_dot = A x + B u + w
  // y = C x
  const int x_dim = 1;
  const int y_dim = 1;
  const int u_dim = 1;
  Eigen::MatrixXd A(x_dim, x_dim);
  Eigen::MatrixXd B(x_dim, u_dim);
  Eigen::MatrixXd C(y_dim, x_dim);
  Eigen::MatrixXd W(x_dim, 1);
  Eigen::MatrixXd x0(x_dim, 1);
  A << -c/m;
  B << 1./m;
  C << 1.0;
  W << 1.0;
  x0 << v0;

  // define the MPC system
  // X = Aex x0 + Bex U + Wex W
  // Y = Cex X
  const int horizon = 30;
  Eigen::MatrixXd Aex(x_dim * horizon, x_dim);
  Eigen::MatrixXd Bex(x_dim * horizon, u_dim * horizon);
  Eigen::MatrixXd Cex(y_dim * horizon, x_dim * horizon);
  Eigen::MatrixXd Wex(x_dim * horizon, 1);
  Eigen::MatrixXd Qex(y_dim * horizon, y_dim * horizon);
  Eigen::MatrixXd Rex(u_dim * horizon, u_dim * horizon);
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(x_dim, x_dim);
  Eigen::MatrixXd Q(y_dim, y_dim);
  Eigen::MatrixXd R(u_dim, u_dim);
  Eigen::MatrixXd w(x_dim, 1);
  Q << 1;
  R << 0.001;
  w << 0.01;

  printf("compute Aex\r\n");
  // compute Aex
  Eigen::MatrixXd last_Aex(x_dim, x_dim);
  last_Aex = I;
  for(int i = 0; i < horizon; i++)
  {
    Eigen::MatrixXd A_pow_i(x_dim, x_dim);
    A_pow_i = last_Aex * A;
    Aex.block<x_dim,x_dim>(i*x_dim, 0) = A_pow_i;
    last_Aex = A_pow_i;
  }

  printf("compute Bex\r\n");
  // compute Bex
  Bex.block<x_dim,u_dim>(0, 0) = B;
  for(int i = 1; i < horizon; i++)
  {
    Bex.block<x_dim,u_dim*horizon>(x_dim*i, 0) = A * Bex.block<x_dim,u_dim*horizon>(i-1, 0);
    Bex.block<x_dim,u_dim>(x_dim*i, u_dim*i) = B;
  }

  printf("compute Wex\r\n");
  // compute Wex
  Eigen::MatrixXd last_Wex(x_dim, 1);
  Wex.block<x_dim,1>(0, 0) = w;
  last_Wex = w;
  for(int i = 1; i < horizon; i++)
  {
    last_Wex = A * last_Wex + w;
    Wex.block<x_dim, 1>(i*x_dim, 0) = last_Wex;
  }

  printf("compute Cex\r\n");
  // compute Cex, Qex, Rex
  for(int i = 0; i < horizon; i++)
  {
    Cex.block<y_dim,x_dim>(i*y_dim, i*x_dim) = C;
    Qex.block<y_dim,y_dim>(i*y_dim, i*y_dim) = Q;
    Rex.block<u_dim,u_dim>(i*u_dim, i*u_dim) = R;
  }

  // 1/2 x^t H x + x^T g
  // s.t. lbA ≦ Ax ≦ ubA
  //      lb ≦ x ≦ ub
  Eigen::MatrixXd H(u_dim * horizon, u_dim * horizon);
  Eigen::MatrixXd g(u_dim * horizon, 1);

  H = 0.5 * Bex.transpose() * Cex.transpose() * Qex * Cex * Bex + Rex;
  g = ( x0.transpose() * Aex.transpose() + Wex.transpose() ) * Cex.transpose() * Qex * Cex * Bex;

  std::cout << "A: " << A << "\n";
  std::cout << "B: " << B << "\n";
  std::cout << "C: " << C << "\n\n";

	USING_NAMESPACE_QPOASES
  real_t lbA[1] = {0};
  real_t ubA[1] = {0};
	real_t const_A[u_dim*horizon] = { 0, 0 };
  real_t lb[u_dim*horizon] = {};
  real_t ub[u_dim*horizon] = {};
  for (int i = 0; i < u_dim*horizon; i++) {
    lb[i] = -10;
    ub[i] = 10;
  }
  real_t H_real[H.size()];
  real_t g_real[g.size()];
  Eigen::Map<Eigen::VectorXd>(H_real, H.size()) = Eigen::Map<const Eigen::VectorXd>(H.data(), H.size());
  Eigen::Map<Eigen::VectorXd>(g_real, g.size()) = Eigen::Map<const Eigen::VectorXd>(g.data(), g.size());
  QProblem qp_solver( u_dim * horizon, 1 );
	Options options;
	qp_solver.setOptions( options );
	/* Solve first QP. */
	int_t nWSR = 50;
	qp_solver.init( H_real, g_real, const_A, lb, ub, lbA, ubA, nWSR );
  /* Get and print solution of first QP. */
	real_t xOpt[u_dim*horizon];
	real_t yOpt[u_dim*horizon+1];
	qp_solver.getPrimalSolution( xOpt );
	qp_solver.getDualSolution( yOpt );
  std::cout << "u = \n";
  std::cout << xOpt[0] << ", "<< xOpt[1] << ", "<< xOpt[2] << ", "<< xOpt[3] << ", "<< xOpt[4] << "\n";
  return 0;
}