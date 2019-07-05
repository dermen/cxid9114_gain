#include <cctbx/boost_python/flex_fwd.h>

#include <cmath>
#include <boost/python/module.hpp>
#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/dict.hpp>
#include <boost/python/list.hpp>
#include <boost/python/return_internal_reference.hpp>

#include <scitbx/array_family/flex_types.h>
#include <scitbx/array_family/shared.h>
#include <scitbx/array_family/boost_python/shared_wrapper.h>
#include <scitbx/examples/bevington/prototype_core.h>

#include <Eigen/Sparse>
#include <Eigen/SparseQR>
#include <Eigen/SparseLU>
#include <Eigen/OrderingMethods>
#include <Eigen/IterativeLinearSolvers>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//#include <scitbx/math/mean_and_variance.h>

using namespace boost::python;
namespace cxid9114{
namespace solvers{

void hello( int n){
    std::cout << "Hello world" << " " << n << std::endl;
    }

typedef scitbx::af::shared<double> vecd;
typedef scitbx::af::shared<size_t> veci;


// LBFGS tools:
vecd grad_vecs_cpp( vecd resid,
                    vecd Gcurr,
                    vecd IAcurr,
                    vecd IBcurr,
                    veci Aidx,
                    veci Gidx,
                    vecd PA,
                    vecd PB,
                    vecd LA,
                    vecd LB, std::size_t Nhkl, std::size_t n ){

    //std::size_t n = Gcurr.size() + IAcurr.size() + IBcurr.size();
    //std::size_t Nhkl = IAcurr.size();
    std::size_t Nmeas = Aidx.size();

    double* resid_ptr = &resid[0];
    double* Gcurr_pt = &Gcurr[0];
    double* IAcurr_pt = &IAcurr[0];
    double* IBcurr_pt = &IBcurr[0];
    double* PA_pt = &PA[0];
    double* PB_pt = &PB[0];
    double* LA_pt = &LA[0];
    double* LB_pt = &LB[0];

    size_t* Aidx_pt = &Aidx[0];
    size_t* Gidx_pt = &Gidx[0];

    vecd grad_vec(n);
    double* grad_vec_ptr = &grad_vec[0];

    for (int i_meas=0; i_meas < Nmeas; i_meas++){

        size_t i_hkl = Aidx_pt[i_meas];
        size_t i_s = Gidx_pt[i_meas];

        grad_vec_ptr[i_hkl] += -resid_ptr[i_meas] * Gcurr_pt[i_meas] * PA_pt[i_meas] * LA_pt[i_meas] * std::exp(IAcurr_pt[i_meas]);

        grad_vec_ptr[Nhkl + i_hkl] += -resid_ptr[i_meas] * Gcurr_pt[i_meas] * PB_pt[i_meas] * LB_pt[i_meas] * std::exp(IBcurr_pt[i_meas]);

        grad_vec_ptr[2*Nhkl + i_s] += -resid_ptr[i_meas] * ( std::exp(IAcurr_pt[i_meas])*PA_pt[i_meas] * LA_pt[i_meas]
                                                + std::exp(IBcurr_pt[i_meas]) * PB_pt[i_meas] * LB_pt[i_meas]);

        }

    return grad_vec;
    }


vecd grad_vecs_cpp2( vecd resid,
                    vecd Gcurr,
                    vecd IAcurr,
                    vecd IBcurr,
                    veci Aidx,
                    veci Gidx,
                    vecd PA,
                    vecd PB,
                    vecd LA,
                    vecd LB, std::size_t Nhkl, std::size_t n ){

    //std::size_t n = Gcurr.size() + IAcurr.size() + IBcurr.size();
    //std::size_t Nhkl = IAcurr.size();
    std::size_t Nmeas = Aidx.size();

    double* resid_ptr = &resid[0];
    double* Gcurr_pt = &Gcurr[0];
    double* IAcurr_pt = &IAcurr[0];
    double* IBcurr_pt = &IBcurr[0];
    double* PA_pt = &PA[0];
    double* PB_pt = &PB[0];
    double* LA_pt = &LA[0];
    double* LB_pt = &LB[0];

    size_t* Aidx_pt = &Aidx[0];
    size_t* Gidx_pt = &Gidx[0];

    vecd grad_vec(n);
    double* grad_vec_ptr = &grad_vec[0];

    for (int i_meas=0; i_meas < Nmeas; i_meas++){

        size_t i_hkl = Aidx_pt[i_meas];
        size_t i_s = Gidx_pt[i_meas];

        grad_vec_ptr[i_hkl] += -resid_ptr[i_meas] * std::exp(Gcurr_pt[i_meas]) * PA_pt[i_meas] * LA_pt[i_meas] * std::exp(IAcurr_pt[i_meas]);

        grad_vec_ptr[Nhkl + i_hkl] += -resid_ptr[i_meas] * std::exp(Gcurr_pt[i_meas]) * PB_pt[i_meas] * LB_pt[i_meas] * std::exp(IBcurr_pt[i_meas]);

        grad_vec_ptr[2*Nhkl + i_s] += -resid_ptr[i_meas] * ( std::exp(IAcurr_pt[i_meas])*PA_pt[i_meas] * LA_pt[i_meas]
                                                + std::exp(IBcurr_pt[i_meas]) * PB_pt[i_meas] * LB_pt[i_meas]) * std::exp(Gcurr_pt[i_meas]);

        }

    return grad_vec;
    }

vecd curvatures_old( vecd resid,
                    vecd Gcurr,
                    vecd IAcurr,
                    vecd IBcurr,
                    veci Aidx,
                    veci Gidx,
                    vecd PA,
                    vecd PB,
                    vecd LA,
                    vecd LB, std::size_t Nhkl, std::size_t n ){

    //std::size_t n = Gcurr.size() + IAcurr.size() + IBcurr.size();
    //std::size_t Nhkl = IAcurr.size();
    std::size_t Nmeas = Aidx.size();

    double* resid_ptr = &resid[0];
    double* Gcurr_pt = &Gcurr[0];
    double* IAcurr_pt = &IAcurr[0];
    double* IBcurr_pt = &IBcurr[0];
    double* PA_pt = &PA[0];
    double* PB_pt = &PB[0];
    double* LA_pt = &LA[0];
    double* LB_pt = &LB[0];

    size_t* Aidx_pt = &Aidx[0];
    size_t* Gidx_pt = &Gidx[0];


    vecd curva(n);
    double* curva_ptr = &curva[0];

    for (int i_meas=0; i_meas < Nmeas; i_meas++){

        size_t i_hkl = Aidx_pt[i_meas];
        size_t i_s = Gidx_pt[i_meas];

        double G = std::exp(Gcurr_pt[i_meas]);
        double A = std::exp(IAcurr_pt[i_meas]);
        double B = std::exp(IBcurr_pt[i_meas]);
        double a = PA_pt[i_meas] * LA_pt[i_meas];
        double b = PB_pt[i_meas] * LB_pt[i_meas];

        double Aterm = A*a*G;
        double Bterm = B*b*G;
        double Gterm = Aterm + Bterm;

        curva_ptr[i_hkl] += Aterm*(Aterm - resid_ptr[i_meas]);
        curva_ptr[Nhkl + i_hkl] += Bterm*(Bterm - resid_ptr[i_meas]);
        curva_ptr[2*Nhkl + i_s] += Gterm* (Gterm - resid_ptr[i_meas]);

        }

    return curva;
    }

vecd curvatures2( vecd resid,
                    vecd Gcurr,
                    vecd IAcurr,
                    vecd IBcurr,
                    veci Aidx,
                    veci Gidx,
                    vecd PA,
                    vecd PB,
                    vecd LA,
                    vecd LB, std::size_t Nhkl, std::size_t n ){

    //std::size_t n = Gcurr.size() + IAcurr.size() + IBcurr.size();
    //std::size_t Nhkl = IAcurr.size();
    std::size_t Nmeas = Aidx.size();

    double* resid_ptr = &resid[0];
    double* Gcurr_pt = &Gcurr[0];
    double* IAcurr_pt = &IAcurr[0];
    double* IBcurr_pt = &IBcurr[0];
    double* PA_pt = &PA[0];
    double* PB_pt = &PB[0];
    double* LA_pt = &LA[0];
    double* LB_pt = &LB[0];

    size_t* Aidx_pt = &Aidx[0];
    size_t* Gidx_pt = &Gidx[0];

    vecd curva(n);
    double* curva_ptr = &curva[0];

    for (int i_meas=0; i_meas < Nmeas; i_meas++){

        size_t i_hkl = Aidx_pt[i_meas];
        size_t i_s = Gidx_pt[i_meas];

        double G = Gcurr_pt[i_meas];
        double A = std::exp(IAcurr_pt[i_meas]);
        double B = std::exp(IBcurr_pt[i_meas]);
        double a = PA_pt[i_meas] * LA_pt[i_meas];
        double b = PB_pt[i_meas] * LB_pt[i_meas];

        double Aterm = A*a;
        double Bterm = B*b;

        curva_ptr[i_hkl] += Aterm*G*(Aterm*G - resid_ptr[i_meas]);
        curva_ptr[Nhkl + i_hkl] += Bterm*G*(Bterm*G - resid_ptr[i_meas]);
        curva_ptr[2*Nhkl + i_s] += (Aterm+Bterm)*(Aterm+Bterm);

        }

    return curva;
    }
/*
~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~
               END LBFGS tool
~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~
*/


// Sparse lev-mar tools

class log_sparse_jac_base: public scitbx::example::non_linear_ls_eigen_wrapper {
  public:
    log_sparse_jac_base(int n_parameters):
      non_linear_ls_eigen_wrapper(n_parameters){}
    // SET A SOLVE METHOD TO OVERRIDE THE CURRENT ONE

    void set_cpp_data(vecd y_obs_, vecd w_obs_,
                    veci Aidx_,
                    veci Gidx_,
                    vecd PA_,
                    vecd PB_,
                    vecd LA_,
                    vecd LB_,
                    std::size_t Nhkl_,
                    std::size_t Ns_){
        y_obs=y_obs_;
        w_obs=w_obs_;
        Aidx=Aidx_;Gidx=Gidx_;
        LA=LA_;LB=LB_;PA=PA_;PB=PB_;
        Nhkl=Nhkl_;Ns=Ns_;
        }


    void set_karl_data(vecd y_obs_, vecd w_obs_,
                veci Aidx_,
                veci Gidx_,
                vecd PA_,
                vecd PB_,
                vecd LA_,
                vecd LB_,
                vecd EN_,
                //vecd a_enA_,
                //vecd a_enB_,
                //vecd b_enA_,
                //vecd b_enB_,
                //vecd c_enA_,
                //vecd c_enB_,
                std::size_t Nhkl_,
                std::size_t Ns_){
        y_obs=y_obs_;
        w_obs=w_obs_;
        Aidx=Aidx_;Gidx=Gidx_;
        LA=LA_;LB=LB_;PA=PA_;PB=PB_;
        EN=EN_;
        //a_enA=a_enA_;
        //a_enB=a_enB_;

        //b_enA=b_enA_;
        //b_enB=b_enB_;

        //c_enA=c_enA_;
        //c_enB=c_enB_;
        Nhkl=Nhkl_;Ns=Ns_;
        }

    vecd fvec_karl(vecd current_values) {
          vecd y_diff = vecd(y_obs.size());
          for (int i = 0; i < y_obs.size(); ++i){

            std::size_t i_hkl = Aidx[i];
            std::size_t i_s = Gidx[i];

            double Iprot = std::exp(current_values[i_hkl]);
            double Iheav = std::exp(current_values[Nhkl + i_hkl]);
            double alpha = current_values[2*Nhkl + i_hkl];
            double G = current_values[3*Nhkl + i_s];

            double COS = std::cos(alpha);
            double SIN = std::sin(alpha);

            double Ipp = Iprot*Iprot;
            double Ihh = Iheav*Iheav;
            double IphCOS = Iprot*Iheav*COS;
            double IphSIN = Iprot*Iheav*SIN;

            //if (i==42)
            //    std::cout << Ipp << " " << Ihh << " " << IphCOS << " " << IphSIN << "\n";

            double a_enA = EN[i_hkl];
            double b_enA = EN[i_hkl + Nhkl];
            double c_enA = EN[i_hkl + 2*Nhkl];
            double a_enB = EN[i_hkl + 3*Nhkl];
            double b_enB = EN[i_hkl + 4*Nhkl];
            double c_enB = EN[i_hkl + 5*Nhkl];

            double karl_A = Ipp + Ihh*a_enA + IphCOS*b_enA + IphSIN*c_enA;
            double karl_B = Ipp + Ihh*a_enB + IphCOS*b_enB + IphSIN*c_enB;

            double y_calc = G * (karl_A * LA[i]*PA[i]
                                        + karl_B*LB[i]*PB[i]);
            y_diff[i] = y_obs[i] - y_calc;
          }
          return y_diff;
        }

    void karlize(bool objective_only,
                 scitbx::af::shared<double> current_values) {

        vecd residuals = fvec_karl(current_values);
        if (objective_only){
          add_residuals(residuals.const_ref(), w_obs.const_ref());
          return;
        }

        veci jacobian_one_row_indices(4);
        vecd jacobian_one_row_data(4);

        size_t* jacobian_one_row_indices_pt = &jacobian_one_row_indices[0];
        double* jacobian_one_row_data_pt = &jacobian_one_row_data[0];

        // add one of the normal equations per each observation
        for (int ix = 0; ix < y_obs.size(); ++ix) {

          std::size_t i_hkl = Aidx[ix];
          std::size_t i_s = Gidx[ix];

          double Iprot = std::exp(current_values[i_hkl]);
          double Iheav = std::exp(current_values[Nhkl + i_hkl]);
          double alpha = current_values[2*Nhkl + i_hkl];
          double Gval = current_values[3*Nhkl + i_s];

            double Ipp = Iprot*Iprot;
            double Ihh = Iheav*Iheav;
            double COS = std::cos(alpha);
            double SIN = std::sin(alpha);
            double IphCOS = Iprot*Iheav*COS;
            double IphSIN = Iprot*Iheav*SIN;

            double a_enA = EN[i_hkl];
            double b_enA = EN[i_hkl + Nhkl];
            double c_enA = EN[i_hkl + 2*Nhkl];
            double a_enB = EN[i_hkl + 3*Nhkl];
            double b_enB = EN[i_hkl + 4*Nhkl];
            double c_enB = EN[i_hkl + 5*Nhkl];

          double Aterm = LA[ix]*PA[ix];
          double Bterm = LB[ix]*PB[ix];

          // first derivitive of "yobs - ycalc" w.r.t. Iprot
          double dIprot = Gval*Aterm*(2*Ipp + IphCOS*b_enA + IphSIN*c_enA)
                + Gval*Bterm*(2*Ipp + IphCOS*b_enB + IphSIN*c_enB);
          jacobian_one_row_indices_pt[0]= i_hkl;
          jacobian_one_row_data_pt[0] = dIprot;

          // derivitive w.r.t. Iheav
          double dIheav = Gval*Aterm*(2*Ihh*a_enA + IphCOS*b_enA + IphSIN*c_enA)
                + Gval*Bterm*(2*Ihh*a_enB + IphCOS*b_enB + IphSIN*c_enB);
          jacobian_one_row_indices_pt[1]= Nhkl + i_hkl;
          jacobian_one_row_data_pt[1] = dIheav;

          // derivitive w.r.t. alpha
          double dalpha = Gval*Aterm*(-IphSIN*b_enA + IphCOS*c_enA)
            +Gval*Bterm*(-IphSIN*b_enB + IphCOS*c_enB);
          jacobian_one_row_indices_pt[2]= 2*Nhkl + i_hkl;
          jacobian_one_row_data_pt[2] = dalpha;

          // derivitive w.r.t. G
          double dG = Aterm*(Ipp + Ihh*a_enA + IphCOS*b_enA + IphSIN*c_enA)
            + Bterm*(Ipp + Ihh*a_enB + IphCOS*b_enB + IphSIN*c_enB);
          jacobian_one_row_indices_pt[3]= 3*Nhkl + i_s;
          jacobian_one_row_data_pt[3] = dG;

          //add_equation(residuals[ix], jacobian_one_row.const_ref(), weights[ix]);
          add_residual(-residuals[ix], 1.0);
          add_equation_eigen(residuals[ix], jacobian_one_row_indices.const_ref(), jacobian_one_row_data.const_ref(), 1.);
        }
    }

    double functional_karl(vecd current_values) {
      double result = 0;
      vecd fvec = fvec_karl(current_values);
      for (int i = 0; i < fvec.size(); ++i) {
        result += fvec[i]*fvec[i]*w_obs[i];
      }
      return result;
    }

    vecd fvec_callable(vecd current_values) {
          vecd y_diff = vecd(y_obs.size());
          for (int i = 0; i < y_obs.size(); ++i){

            std::size_t i_hkl = Aidx[i];
            std::size_t i_s = Gidx[i];

            double IAval = std::exp(current_values[i_hkl]);
            double IBval = std::exp(current_values[Nhkl +i_hkl]);
            double Gval = current_values[2*Nhkl + i_s];

            double y_calc = Gval * (IAval * LA[i]*PA[i]
                                        + IBval*LB[i]*PB[i]);
            y_diff[i] = y_obs[i] - y_calc;
          }
          return y_diff;
    }

    void access_cpp_build_up_directly_eigen_eqn(
                            bool objective_only,
                            scitbx::af::shared<double> current_values) {

        vecd residuals = fvec_callable(current_values);
        if (objective_only){
          add_residuals(residuals.const_ref(), w_obs.const_ref());
          return;
        }

        veci jacobian_one_row_indices(3);
        vecd jacobian_one_row_data(3);

        size_t* jacobian_one_row_indices_pt = &jacobian_one_row_indices[0];
        double* jacobian_one_row_data_pt = &jacobian_one_row_data[0];

        // add one of the normal equations per each observation
        for (int ix = 0; ix < y_obs.size(); ++ix) {

          std::size_t i_hkl = Aidx[ix];
          std::size_t i_s = Gidx[ix];

          double IAval = std::exp(current_values[i_hkl]);
          double IBval = std::exp(current_values[Nhkl +i_hkl]);
          double Gval = current_values[2*Nhkl + i_s];

          double Aterm = IAval*LA[ix]*PA[ix];
          double Bterm = IBval * LB[ix]*PB[ix];

          // first derivitive of "yobs - ycalc" w.r.t. IA
          double dIA = Gval * Aterm; //IAval * LA[ix] * PA[ix];
          //jacobian_one_row_indices.push_back( i_hkl );
          //jacobian_one_row_data.push_back(dIA);
          jacobian_one_row_indices_pt[0]= i_hkl;
          jacobian_one_row_data_pt[0] = dIA;

          // derivitive w.r.t. IB
          double dIB = Gval * Bterm;// IBval *LB[ix] * PB[ix];
          //jacobian_one_row_indices.push_back( Nhkl+i_hkl );
          //jacobian_one_row_data.push_back(dIB);
          jacobian_one_row_indices_pt[1]= Nhkl + i_hkl;
          jacobian_one_row_data_pt[1] = dIB;

          // derivitive w.r.t. G
          double dG = Aterm + Bterm; //IAval * LA[ix] * PA[ix]
                //+ IBval * LB[ix] * PB[ix];
          //jacobian_one_row_indices.push_back(2*Nhkl + i_s);
          //jacobian_one_row_data.push_back(dG);
          jacobian_one_row_indices_pt[2]= 2*Nhkl + i_s;
          jacobian_one_row_data_pt[2] = dG;

          //add_equation(residuals[ix], jacobian_one_row.const_ref(), weights[ix]);
          add_residual(-residuals[ix], 1.0);
          add_equation_eigen(residuals[ix], jacobian_one_row_indices.const_ref(), jacobian_one_row_data.const_ref(), 1.);
          }
    }

    double functional(vecd current_values) {
      double result = 0;
      vecd fvec = fvec_callable(current_values);
      for (int i = 0; i < fvec.size(); ++i) {
        result += fvec[i]*fvec[i]*w_obs[i];
      }
      return result;
    }

    vecd y_obs,w_obs,PA,PB,LA,LB;
    //vecd a_enA, a_enB;
    //vecd b_enA, b_enB;
    //vecd c_enA, c_enB
    vecd EN;
    veci Aidx,Gidx;
    std::size_t Nhkl, n, Ns, Ne;

    //veci jacobian_one_row_indices(3);
    //vecd jacobian_one_row_data(3);
  };
// END Sparse lev-mar tools



namespace boost_python {
  namespace {

  void solvers_init_module() {
    using namespace boost::python;

    typedef return_value_policy<return_by_value> rbv;
    typedef default_call_policies dcp;

    def ("hello", &hello, ( arg("n") ) );

    def ("grad_vecs_cpp", &grad_vecs_cpp, (
        arg("resid") ,
        arg("Gcurr") ,
        arg("IAcurr") ,
        arg("IBcurr") ,
        arg("Aidx") ,
        arg("Gidx") ,
        arg("PA") ,
        arg("PB") ,
        arg("LA") ,
        arg("LB"), arg("Nhkl"), arg("n")
        ) );

    def ("grad_vecs_cpp2", &grad_vecs_cpp2, (
        arg("resid") ,
        arg("Gcurr") ,
        arg("IAcurr") ,
        arg("IBcurr") ,
        arg("Aidx") ,
        arg("Gidx") ,
        arg("PA") ,
        arg("PB") ,
        arg("LA") ,
        arg("LB"), arg("Nhkl"), arg("n")
        ) );

    def ("curvatures2", &curvatures2, (
        arg("resid") ,
        arg("Gcurr") ,
        arg("IAcurr") ,
        arg("IBcurr") ,
        arg("Aidx") ,
        arg("Gidx") ,
        arg("PA") ,
        arg("PB") ,
        arg("LA") ,
        arg("LB"), arg("Nhkl"), arg("n")
            )
          );

    typedef scitbx::example::non_linear_ls_eigen_wrapper nllsew ;

    typedef log_sparse_jac_base lsjb ;
    class_<lsjb,bases<nllsew> >( "log_sparse_jac_base", no_init)

      .def( init<int>(arg("n_parameters")))

      .def(
        "access_cpp_build_up_directly_eigen_eqn",
        &lsjb::access_cpp_build_up_directly_eigen_eqn,
        (arg("objective_only"),arg("current_values")))

      .def(
        "fvec_callable",
        &lsjb::fvec_callable,
        (arg("current_values")) )

      .def(
        "functional",
        &lsjb::functional,
        (arg("current_values")) )

      //.def("solve", &lsjb::solve) // no arguments

      .def(
        "set_cpp_data",
        &lsjb::set_cpp_data,
        (arg("y_obs_"), arg("w_obs_"), arg("Aidx_"), arg("Gidx_"), arg("PA_"),
         arg("PB_"), arg("LA_"), arg("LB_"), arg("Nhkl_"),
         arg("Ns_")) )

      .def(
        "functional_karl",
        &lsjb::functional_karl,
        (arg("current_values")) )

      .def(
        "fvec_karl",
        &lsjb::fvec_karl,
        (arg("current_values")) )


      .def(
        "karlize",
        &lsjb::karlize,
        (arg("objective_only"),arg("current_values")))

      .def(
        "set_karl_data",
        &lsjb::set_karl_data,
        (arg("y_obs_"),
         arg("w_obs_"),
         arg("Aidx_"),
         arg("Gidx_"),
         arg("PA_"),
         arg("PB_"),
         arg("LA_"),
         arg("LB_"),
         arg("EN_"),
         arg("Nhkl_"),
         arg("Ns_")) )
      ;



      /*
   class_<sparse_karl,bases<nllsew> >( "sparse_karl", no_init)

      .def( init<int>(arg("n_parameters")))


      .def(
        "set_karl_data",
        &sparse_karl::set_cpp_data,
        (arg("y_obs_"), arg("w_obs_"), arg("Aidx_"), arg("Gidx_"), arg("PA_"),
         arg("PB_"), arg("LA_"), arg("LB_"),
           arg("a_enA_"),
            arg("a_enB_"),
            arg("b_enA_"),
            arg("b_enB_"),
            arg("c_enA_"),
            arg("c_enB_"),
            arg("Nhkl_"),
            arg("Ns_")) )

      .def(
        "functional_karl",
        &lsjb::functional_karl,
        (arg("current_values")) )

      .def(
        "fvec_karl",
        &lsjb::fvec_karl,
        (arg("current_values")) )

      .def(
        "karlize",
        &lsjb::karlize,
        (arg("objective_only"),arg("current_values")))

      ;
      */


  }

}}}
}// namespace xfel::boost_python::<anonymous>

BOOST_PYTHON_MODULE(cxid9114_solvers_ext)
{
  cxid9114::solvers::boost_python::solvers_init_module();

}
