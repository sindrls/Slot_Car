
import numpy as np
import casadi as ca
import scipy


class Example4To3(ca.Callback):
  def __init__(self, name, opts={}):
    ca.Callback.__init__(self)
    self.construct(name, opts)

  def get_n_in(self): return 1
  def get_n_out(self): return 1

  def get_sparsity_in(self,i):
    return ca.Sparsity.dense(4,1)

  def get_sparsity_out(self,i):
    return ca.Sparsity.dense(3,1)

  # Evaluate numerically
  def eval(self, arg):
    a,b,c,d = ca.vertsplit(arg[0])
    ret = ca.vertcat(ca.sin(c)*d+d**2,2*a+c,b**2+5*c)
    return [ret]
class Example4To3_Jac(Example4To3):
  def has_jacobian(self): return True
  def get_jacobian(self,name,inames,onames,opts):
    class JacFun(ca.Callback):
      def __init__(self, opts={}):
        ca.Callback.__init__(self)
        self.construct(name, opts)

      def get_n_in(self): return 2
      def get_n_out(self): return 1

      def get_sparsity_in(self,i):
        if i==0: # nominal input
          return ca.Sparsity.dense(4,1)
        elif i==1: # nominal output
          return ca.Sparsity(3,1)

      def get_sparsity_out(self,i):
        return ca.sparsify(ca.DM([[0,0,1,1],[1,0,1,0],[0,1,1,0]])).sparsity()



      # Evaluate numerically
      def eval(self, arg):

        turn_radius = 0.3

        l_x_vals = np.asarray(
        [0, -turn_radius, -1.5 * turn_radius, - 2.5 * turn_radius, - 1.5 * turn_radius, -1 * turn_radius,
        1.5 * turn_radius, 2 * turn_radius, 3 * turn_radius, 2 * turn_radius, 1 * turn_radius, 2 * turn_radius,
        2.5 * turn_radius,
        3 * turn_radius, 3.5 * turn_radius, 4.5 * turn_radius, 3.5 * turn_radius, 3 * turn_radius, 0])
        l_y_vals = np.asarray(
        [turn_radius, turn_radius, turn_radius, 0, -turn_radius, -turn_radius, -turn_radius, -turn_radius,
        - 2 * turn_radius, -3 * turn_radius, -2 * turn_radius, - turn_radius, -turn_radius, -turn_radius,
        -turn_radius, 0, turn_radius, turn_radius, turn_radius])

        phi_vals = np.asarray([0, turn_radius, 1.5 * turn_radius, 1.5 * turn_radius + (np.pi / 2) * turn_radius,
        1.5 * turn_radius + np.pi * turn_radius, 2 * turn_radius + np.pi * turn_radius,
        4.5 * turn_radius + np.pi * turn_radius, 5 * turn_radius + np.pi * turn_radius,
        1.5 + np.pi * (3.0 / 2.0) * turn_radius, 5 * turn_radius + np.pi * 2 * turn_radius,
        5 * turn_radius + np.pi * (5.0 / 2.0) * turn_radius,
        5 * turn_radius + np.pi * 3 * turn_radius, 5.5 * turn_radius + np.pi * 3 * turn_radius,
        6 * turn_radius + np.pi * 3 * turn_radius, 6.5 * turn_radius + np.pi * 3 * turn_radius,
        6.5 * turn_radius + np.pi * 3.5 * turn_radius,
        6.5 * turn_radius + np.pi * 4 * turn_radius,
        7 * turn_radius + np.pi * 4 * turn_radius, 10 * turn_radius + np.pi * 4 * turn_radius])

        spline_points = np.linspace(0, phi_vals[-1], 1000)

        l_x = scipy.interpolate.CubicSpline(phi_vals, l_x_vals, bc_type='periodic')
        d_phi_l_x = l_x.derivative()
        dd_phi_l_x = d_phi_l_x.derivative()

        l_y = scipy.interpolate.CubicSpline(phi_vals, l_y_vals, bc_type='periodic')
        d_phi_l_y = l_y.derivative()
        dd_phi_l_y = d_phi_l_y.derivative()


        a,b,c,d = ca.vertsplit(arg[0])
        ret = ca.DM(3,4)
        ret[0,2] = d*ca.cos(c)
        ret[0,3] = ca.sin(c)+2*d
        ret[1,0] = dd_phi_l_y(a)
        ret[1,2] = 1
        ret[2,1] = 2*b
        ret[2,2] = 5
        return [ret]

    # You are required to keep a reference alive to the returned Callback object
    self.jac_callback = JacFun()
    return self.jac_callback

f = Example4To3_Jac('f')
x = ca.MX.sym("x",4)
J = ca.Function('J',[x],[ca.jacobian(f(x),x)])
print(J(ca.vertcat(1,2,0,3)))


