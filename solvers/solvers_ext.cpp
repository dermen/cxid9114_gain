#include <cctbx/boost_python/flex_fwd.h>

#include <boost/python/module.hpp>
#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/dict.hpp>
#include <boost/python/list.hpp>
#include <scitbx/array_family/flex_types.h>
#include <scitbx/array_family/shared.h>
#include <cmath>
#include <scitbx/array_family/boost_python/shared_wrapper.h>
#include <Eigen/Sparse>
#include <boost/python/return_internal_reference.hpp>

using namespace boost::python;
namespace cxid9114{
namespace solvers{

void hello( int n){
    std::cout << "Hello world" << " " << n << std::endl;
    }

typedef scitbx::af::shared<double> vecd;
typedef scitbx::af::shared<size_t> veci;

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



namespace boost_python { namespace {

  void
  solvers_init_module() {
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
        arg("LB"), ("Nhkl"), ("n")
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
        arg("LB"), ("Nhkl"), ("n")
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
        arg("LB"), ("Nhkl"), ("n")
        ) );

  }

}
}}} // namespace xfel::boost_python::<anonymous>

BOOST_PYTHON_MODULE(cxid9114_solvers_ext)
{
  cxid9114::solvers::boost_python::solvers_init_module();

}
