#include "ros/ros.h"
#include <iostream>
#include <Eigen/Core>
#include <qpOASES.hpp>
#include "std_msgs/Float64.h"

class System
{
public:
  System(const int x_dim, const int y_dim, const int u_dim)
   : x_dim(x_dim), y_dim(y_dim), u_dim(u_dim), A(x_dim, x_dim), B(x_dim, u_dim), C(y_dim, x_dim), W(x_dim, 1)
  {
  }
  // Set System
  // x_dot = A x + B u
  // y = C x
  Eigen::MatrixXd A;
  Eigen::MatrixXd B;
  Eigen::MatrixXd C;
  Eigen::MatrixXd W;
  const int x_dim;
  const int y_dim;
  const int u_dim;
};

class MPC
{
public:
  MPC(System sys, const int horizon)
   :  system_(sys), 
      horizon_(horizon),
      Aex(sys.x_dim * horizon, sys.x_dim), 
      Bex(sys.x_dim * horizon, sys.u_dim * horizon), 
      Cex(sys.y_dim * horizon, sys.x_dim * horizon), 
      Wex(sys.x_dim * horizon, 1), 
      Qex(sys.y_dim * horizon, sys.y_dim * horizon), 
      Rex(sys.u_dim * horizon, sys.u_dim * horizon), 
      Q(sys.y_dim, sys.y_dim), 
      R(sys.u_dim, sys.u_dim), 
      w(sys.x_dim, 1),
      x_dim(sys.x_dim), 
      y_dim(sys.y_dim), 
      u_dim(sys.u_dim)
  {
    Q << 10;
    R << 0.1;
    w << 0.0, 0.0;
    initializeMatrix();
  }

  void initializeMatrix()
  {
    printf("compute Aex\r\n");
    // compute Aex
    Eigen::MatrixXd last_Aex(x_dim, x_dim);
    last_Aex = Eigen::MatrixXd::Identity(x_dim, x_dim);;
    for(int i = 0; i < horizon_; i++)
    {
      Eigen::MatrixXd A_pow_i(x_dim, x_dim);
      A_pow_i = last_Aex * system_.A;
      Aex.block(x_dim * i, 0, x_dim, x_dim) = A_pow_i;
      last_Aex = A_pow_i;
    }

    printf("compute Bex\r\n");
    // compute Bex
    Bex.block(0, 0, x_dim, u_dim) = system_.B;
    for(int i = 1; i < horizon_; i++)
    {
      Bex.block(x_dim*i, 0, x_dim, u_dim*horizon_) = system_.A * Bex.block(i-1, 0, x_dim, u_dim*horizon_);
      Bex.block(x_dim*i, u_dim*i, x_dim, u_dim) = system_.B;
    }

    printf("compute Wex\r\n");
    // compute Wex
    Eigen::MatrixXd last_Wex(x_dim, 1);
    Wex.block(0, 0, x_dim, 1) = w;
    last_Wex = w;
    for(int i = 1; i < horizon_; i++)
    {
      last_Wex = system_.A * last_Wex + w;
      Wex.block(i*x_dim, 0, x_dim, 1) = last_Wex;
    }

    printf("compute Cex\r\n");
    // compute Cex, Qex, Rex
    for(int i = 0; i < horizon_; i++)
    {
      Cex.block(i*y_dim, i*x_dim, y_dim, x_dim) = system_.C;
      Qex.block(i*y_dim, i*y_dim, y_dim, y_dim) = Q;
      Rex.block(i*u_dim, i*u_dim, u_dim, u_dim) = R;
    }
  }

  double solveQP(Eigen::MatrixXd x0, Eigen::MatrixXd ref)
  {
    // 1/2 x^t H x + x^T g
    // s.t. lbA ≦ Ax ≦ ubA
    //      lb ≦ x ≦ ub
    Eigen::MatrixXd H(u_dim * horizon_, u_dim * horizon_);
    Eigen::MatrixXd g(u_dim * horizon_, 1);

    Eigen::MatrixXd Refex(y_dim * horizon_, 1);
    for(int i = 0; i < horizon_; i++)
    {
      Refex.block(y_dim*i, 0, y_dim, 1) = ref;
    }

    H = 2 * Bex.transpose() * Cex.transpose() * Qex * Cex * Bex + Rex;
    //g = (( x0.transpose() * Aex.transpose() + Wex.transpose() ) * Cex.transpose() - Refex.transpose()) * Qex * Cex * Bex;
    g = 2 * (( Cex*(Aex * x0 + Wex) - Refex ).transpose() * Qex * Cex * Bex);

    USING_NAMESPACE_QPOASES
    
    real_t lbA[1] = {0};
    real_t ubA[1] = {0};
    real_t const_A[u_dim*horizon_] = { 0, 0 };
    real_t lb[u_dim*horizon_] = {};
    real_t ub[u_dim*horizon_] = {};
    for (int i = 0; i < u_dim*horizon_; i++) {
      lb[i] = -10;
      ub[i] = 10;
    }
    real_t H_real[H.size()];
    real_t g_real[g.size()];
    Eigen::Map<Eigen::VectorXd>(H_real, H.size()) = Eigen::Map<const Eigen::VectorXd>(H.data(), H.size());
    Eigen::Map<Eigen::VectorXd>(g_real, g.size()) = Eigen::Map<const Eigen::VectorXd>(g.data(), g.size());
    QProblem qp_solver( u_dim * horizon_, 1 );
    Options options;
    qp_solver.setOptions( options );
    /* Solve first QP. */
    int_t nWSR = 50;
    qp_solver.init( H_real, g_real, const_A, lb, ub, lbA, ubA, nWSR );
    /* Get and print solution of first QP. */
    real_t xOpt[u_dim*horizon_];
    real_t yOpt[u_dim*horizon_+1];
    qp_solver.getPrimalSolution( xOpt );
    qp_solver.getDualSolution( yOpt );

    return xOpt[0];
  }

private:
  // define the MPC system
  // X = Aex x0 + Bex U + Wex W
  // Y = Cex X
  const int horizon_;
  System system_;
  Eigen::MatrixXd Aex;
  Eigen::MatrixXd Bex;
  Eigen::MatrixXd Cex;
  Eigen::MatrixXd Wex;
  Eigen::MatrixXd Qex;
  Eigen::MatrixXd Rex;
  Eigen::MatrixXd Q;
  Eigen::MatrixXd R;
  Eigen::MatrixXd w;
  const int x_dim;
  const int y_dim;
  const int u_dim;
};

double system_x = 0;
double system_v = 0;
void xCallback(const std_msgs::Float64::ConstPtr& msg)
{
  system_x = msg->data;
}
void vCallback(const std_msgs::Float64::ConstPtr& msg)
{
  system_v = msg->data;
}

int main(int argc, char **argv)
{
  // Define the first-order system
  // m x_2dot + c x_dot + k x = u
  // m: mass
  // c: friction
  // k: sping
  // u: input force
  double m = 1.0; // mass [kg]
  double c = 0.1; // friction [N/(m/s)]
  double k = 0.5; // spring [N/m]
  double x_0 = 0.0;  // initial position[m]
  double v_0 = 0.0;  // initial speed[m/s]
  double T = 0.02;  // Discretization cycle (mpc cycle)

  const int x_dim = 2;
  const int y_dim = 1;
  const int u_dim = 1;
  // Define the linear system (2nd-order)
  // already discretized
  System second_order_system(x_dim, y_dim, u_dim);
  second_order_system.A << 1, T, 
                           -m/k*T, 1-c/m*T;
  second_order_system.B << 0, 
                          T/m;
  second_order_system.C << 1.0, 0.0;
  second_order_system.W << 0.0,  
                            0.0;
  Eigen::MatrixXd x0(x_dim, 1);
  Eigen::MatrixXd ref(y_dim, 1);
  x0 << x_0, 
        v_0;
  ref << 0.2;
  const int horizon = 30;
  MPC mpc(second_order_system, horizon);
  mpc.solveQP(x0, ref);

  ros::init(argc, argv, "mpc_node");
  ros::NodeHandle n;
  ros::Publisher u_pub = n.advertise<std_msgs::Float64>("input_force", 1000);
  ros::Subscriber v_sub = n.subscribe("speed", 1000, vCallback);
  ros::Subscriber x_sub = n.subscribe("position", 1000, xCallback);
  ros::Rate loop_rate(50);
  while (ros::ok())
  {
    x0 << system_x, system_v;
    double u = mpc.solveQP(x0, ref);
    std_msgs::Float64 msg;
    msg.data = u;
    ROS_INFO("u = %f", msg.data);
    u_pub.publish(msg);
    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}