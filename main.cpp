//This code serves to verify the JMPS model, see also
//M. Vasoya, B. Kondori and A.A. Benzerga et al./Journal of the Mechanics and Physics of Solids 136 (2020) 103699
//----------Define modules & Header Files---------//
#include <dolfin.h>
#include <cmath>
#include <math.h>
#include <string>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <random>
#include <stdlib.h> 
#include <algorithm>

#include "scalarfunc.h"     //finiteelement scalar        //energy
#include "tfunc2by2.h"      //tensorelement shape = (2,2) //stress,strain
#include "tfunc3by3.h"      //tensorelement shape = (3,3) //H,D,L in vogit notation
#include "vf.h"             //weak form                   //displacement
#include "project.h"        //compute image field strain from image field displacement using projection
#include "visual_eps.h"     //visualize the strain tensor on finer mesh
#include "visual_sts.h"     //visualize the stress tensor on finer mesh
#include "sts_boundary.h"   //compute shear stress at top boundary

using namespace dolfin;
//------------------------------------------------//
//----------Define Problem Parameters-------------//
//math
#define pi_  3.14159265358979323846

//user-defined tolerance
#define distol			0
#define dist_tol        1e-10

//geometry information (unit:m) -> *10^9 -> (unit:nm)
#define L       64        //nm  //length of domain
#define h       256       //nm  //height of domain
#define x0      0.0       //nm  //left-bottom node x coordinate
#define y0      0.0       //nm  //right-bottom node y coordinate
#define x1      x0 + L    //nm  //right-top node x coordinate
#define y1      y0 + h    //nm  //right-top node y coordinate
#define nx      20 * 4    //-  //number of elements in x dir //NOT specified -> keep the same as 2021-paper
#define ny      60 * 4    //-  //number of elements in y dir //NOT specified -> keep the same as 2021-paper

//material properties
#define E       100                     //nN/nm^2  //Young's Modulus 100e9 Pa //Pa = N / m^2 = (10 ^ -9) kg/(nm * s^2)
#define nu      0.325                   //--  //Possion's Ratio
#define radius  0.6                     //nm   //STZ radius 0.6e-9 m
#define B_      0.001                   //s   //characteristic time for the transformation //??? find specific value //same in Manish's code
#define G       E / ( 2 * ( 1 + nu ) )  //nN/nm^2 //shear modulus
#define phi_c   0.02                   //GJ/m^3 //mean shear strain energy (test with MJ/m^3) //MJ/m^3 = (1e-3) kg/(nm * s^2)
#define phi_s   0.025 * phi_c            //GJ/m^3 //standard devidation for shear strain energy (test with MJ/m^3) 
#define phi_threshold phi_s * 2

//boundary condition
#define gamma_dot  0.1                  //1/s //strain rate //this is engineering strain: 2 * epsilon_12s
#define gamma_threshold 0.093           //-- //threshold strain value //from the figure, there is maximum 3 reactivations for each stz, the simulation time actually doesn't allow more than 3 reactivation in this examplae
#define gamma_threshold_per_iter 0.031  //threshold strain value for each time activation of STZ

#define epsilon_threshold 0.093           //-- //threshold strain value //from the figure, there is maximum 3 reactivations for each stz, the simulation time actually doesn't allow more than 3 reactivation in this examplae
#define epsilon_threshold_per_iter 0.031  //threshold strain value for each time activation of STZ

//Time stepping
#define dt       1e-3     //s  //time stepping //NOT specified > 4.61e-3 to avoid resolve the transformation 
#define T        3e-1     //s  //total simulation time //0.6 -> gives overall shear strain 0.06 
double dt_small = 5e-5;   //s  //NOT specified
int dt_flag  = 1;         //flag: 1->dt 2->dt_small

//STZ locs/shear strain energy pseudo-number generator
int ij = 12; //seed 1
int kl = 67; //seed 2
int N0 = 2200; //initial number of stzs
int N_new = 2200; //current number of stzs

//parameters for pesudo-number generator for stz positions
static double randU[97], randC, randCD, randCM;
static int i97,j97;
static int test = false;

//number of processors
int rank_num = 4;

//eigenstrain option
int option = 3; //1: instantaneous growth (if dt > 4.6e-3) 2: resolve growth history (if dt < 4.6e-3)

//reactivation option
#define t_terminate   4.61 * B_           //s termination time for the STZ to complete transformation //-> dt should be greater than 4.61e-3 in order to allow transformation to be completed within one time step
double param_rho = 3;                    //- scalar as multiply of termination time ( param_rho = 3, 5, 10)
double t_d = param_rho * t_terminate;     //s reactivation time
bool reactivation = true;

//double step_eigenstrain = gamma_threshold_per_iter / ( t_terminate / dt_small );


//------------------------------------------------//
//----------User Defined Nonlinear Problem--------//
// User defined nonlinear problem
class NSequation : public NonlinearProblem
{
	public:

	// Constructor
	NSequation(std::shared_ptr<const Form> F, std::shared_ptr<const Form> J, std::vector<DirichletBC*> bcs) : _F(F), _J(J), _bcs(bcs) {}

	// User defined residual vector
	void F(GenericVector& b, const GenericVector& x)
	{
		assemble(b, *_F);
		for (std::size_t i = 0; i < _bcs.size(); i++)
			_bcs[i]->apply(b, x);
	}

	// User defined assemble of Jacobian
	void J(GenericMatrix& A, const GenericVector& x)
	{
		assemble(A, *_J);
		for (std::size_t i = 0; i < _bcs.size(); i++)
			_bcs[i]->apply(A);
	}

	private:

	// Forms
	std::shared_ptr<const Form> _F;
	std::shared_ptr<const Form> _J;
	std::vector<DirichletBC*> _bcs;
};
//----------Define Problem Boundaries-------------//
// Define top boundary
class BC_top : public SubDomain
{
    bool inside(const Array<double>& x, bool on_boundary) const
    {
        //return x[1] > y1 - DOLFIN_EPS;
        return x[1] > y1 - dist_tol; //change due to domain scaling
    }
};

// Define bottom boundary
class BC_bottom : public SubDomain
{
    bool inside(const Array<double>& x, bool on_boundary) const
    {
        return x[1] < 0.0 + dist_tol;
    }
};

// Define left boundary
class BC_left : public SubDomain
{
    bool inside(const Array<double>& x, bool on_boundary) const
    {
        return x[0] < 0.0 + dist_tol;
    }
};

// Define right boundary
class BC_right : public SubDomain
{
    bool inside(const Array<double>& x, bool on_boundary) const
    {
        //return x[0] > x1 - DOLFIN_EPS;
        return x[0] > x1 - dist_tol; //change due to domain scaling
    }
};

//Define point boundary
class BC_point : public SubDomain
{
    bool inside(const Array<double>& x, bool on_boundary) const
    {
        //return ( x[1] < 0.0 + 1e-5 ) && ( x[0] < 0.0 + 1e-5 ); 
        return near(x[1],y0,dist_tol) && near(x[0],x0 + L/2,dist_tol);
    }
};
//------------------------------------------------//
//----------Define Simple Shear Boundary----------//
//This class now is packed with tilde displacement field at the top boundary
//evaluated at x direction
class simpleshear : public Expression
{
    public:
		
	simpleshear(std::shared_ptr<Function> u_tilde) : _tcur(0) {
        _u_tilde = u_tilde;
    } //This tcur_ is not automatically updated
		
    void eval(Array<double>& values, const Array<double>& x) const
	{
        Array<double> point_value(2); 
        Array<double> point_locs(2);
        point_locs[0] = x[0];
        point_locs[1] = x[1];
        _u_tilde->eval(point_value,point_locs);

		values[0] = h * gamma_dot * _tcur - point_value[0];
	}
	
    // Current time
	double _tcur;
    std::shared_ptr<Function> _u_tilde;
};

//----------Define Uniaxial Tension Boundary-----------//
//This class is included with tilde displacement field at the top boundary
//evaluated at y direction
class axialtension : public Expression
{
    public:

    axialtension(std::shared_ptr<Function> u_tilde) : _tcur(0) {
        _u_tilde = u_tilde;
    } //This tcur_ is not automatically updated
		
    void eval(Array<double>& values, const Array<double>& x) const
	{
        Array<double> point_value(2); 
        Array<double> point_locs(2);
        point_locs[0] = x[0];
        point_locs[1] = x[1];
        _u_tilde->eval(point_value,point_locs);

		values[0] = h * gamma_dot * _tcur - point_value[1];
	}
	
    // Current time
	double _tcur;
    std::shared_ptr<Function> _u_tilde;

};

//----------Define Zero Displs Boundary x-dir----------//
class zero_scalarx : public Expression
{
    public:
    
    zero_scalarx(std::shared_ptr<Function> u_tilde){
         _u_tilde = u_tilde;
    }

    void eval(Array<double>& values, const Array<double>& x) const
    {
        Array<double> point_value(2); 
        Array<double> point_locs(2);
        point_locs[0] = x[0];
        point_locs[1] = x[1];
        _u_tilde->eval(point_value,point_locs);

        values[0] = -1 * point_value[0];
    }

    std::shared_ptr<Function> _u_tilde;
};

//----------Define Zero Displs Boundary y-dir----------//
class zero_scalary : public Expression
{
    public:
    
    zero_scalary(std::shared_ptr<Function> u_tilde){
         _u_tilde = u_tilde;
    }

    void eval(Array<double>& values, const Array<double>& x) const
    {
        Array<double> point_value(2); 
        Array<double> point_locs(2);
        point_locs[0] = x[0];
        point_locs[1] = x[1];
        _u_tilde->eval(point_value,point_locs);

        values[0] = -1 * point_value[1];
    }

    std::shared_ptr<Function> _u_tilde;
};
//------------------------------------------------//
//----------Define Periodic Boundary--------------//
class PeriodBC : public SubDomain
{   
    //target boundary: left boundary
    bool inside(const Array<double>& x, bool on_boundary) const
    {
        //return (near(x[0],x0,distol));
        return x[0] < 0.0 + DOLFIN_EPS;
    }

    //map right boundary to left boundary
    void map(const Array<double>& x, Array<double>& y) const
    {
        y[0] = x[0] - L;
        y[1] = x[1];
    }
};
//------------------------------------------------//
//----------Define Eshelby Analytic Solution------//
class Eshelby
{   

    public:

    //Compute lambda(relative position to STZ center)
    double lambdaxy(double yM[2]) const
    {
        if (sqrt(yM[0] * yM[0] + yM[1] * yM[1]) <= radius){
            return 0;
        }
        else{
            return yM[0] * yM[0] + yM[1] * yM[1] - radius * radius;
        }

    }

    //Compute Invariants
    //dim 1
    double Invt_d1(int i, double yM[2]) const
    {
        if ( i == 1 || i == 2 ){
            return ( 2 * pi_ * radius * radius ) / ( radius * radius + lambdaxy(yM) );
        }
        else{
            std::cout << "INVT_D1: INVALID INDEX" << std::endl;
            return 0;
        }
    }

    //dim 2
    double Invt_d2(int i, int j, double yM[2]) const
    {
        if( ( i == 1 && j == 1 ) || ( i == 2 && j == 2 ) ){
            return ( pi_ * radius * radius ) / ( 3 * pow( radius * radius + lambdaxy(yM) , 2 ) ); 
        }
        else if( ( i == 1 && j == 2 ) || ( i == 2 && j == 1 ) ){
            return ( pi_ * radius * radius ) / ( 1 * pow( radius * radius + lambdaxy(yM) , 2 ) ); 
        }
        else{
            std::cout << "INVT_D2: INVALID INDEX" << std::endl;
            return 0;
        }
    }

    //Compute Spatial Derivatives of Invariants
    //1st derivative
    //dim 1
    double DInvt_d1(int i, int j, double yM[2]) const
    {
        if ( lambdaxy(yM) != 0 ){ //to compute derivative, lambda cannot be zero
            if ( j == 1 ){
                return ( -4 * radius * radius * pi_ * yM[0] ) / pow( yM[0] * yM[0] + yM[1] * yM[1] , 2);
            }
            else if ( j == 2 ){
                return ( -4 * radius * radius * pi_ * yM[1] ) / pow( yM[0] * yM[0] + yM[1] * yM[1] , 2);
            }
            else{
                std::cout << "DINVT_D1: INVALID INDEX" << std::endl;
                return 0;
            }
        }
        else{
            return 0.0;
        }
    }

    //dim 2
    double DInvt_d2(int i, int j, int k, double yM[2]) const
    {
        if ( lambdaxy(yM) != 0 ){
            if( ( i == 1 && j == 1 ) || ( i == 2 && j == 2 ) ){
                if ( k == 1 ){
                    return ( - 4 * radius * radius * pi_ * yM[0] ) / ( 3 * pow( yM[0] * yM[0] + yM[1] * yM[1], 3 ) );
                } 
                else if ( k == 2 ){
                    return ( - 4 * radius * radius * pi_ * yM[1] ) / ( 3 * pow( yM[0] * yM[0] + yM[1] * yM[1], 3 ) );
                }
                else{
                    std::cout << "DINVT_D2: k1: INVALID INDEX" << std::endl;
                    return 0;
                }
            }
            else if( ( i == 1 && j == 2 ) || ( i == 2 && j == 1 ) ){
                if ( k == 1 ){
                    return ( - 4 * radius * radius * pi_ * yM[0] ) / ( 1 * pow( yM[0] * yM[0] + yM[1] * yM[1], 3 ) );
                }
                else if ( k == 2 ){
                    return ( - 4 * radius * radius * pi_ * yM[1] ) / ( 1 * pow( yM[0] * yM[0] + yM[1] * yM[1], 3 ) );
                }
                else{
                    std::cout << "DINVT_D2: k2: INVALID INDEX" << std::endl;
                    return 0;
                }
            }
            else{
                std::cout << "DINVT_D2: ij: INVALID INDEX" << std::endl;
                return 0;
            }
        }
        else{
            return 0.0;
        }
    }

    //2nd derivative
    //dim 1
    double DDInvt_d1(int i, int j, int k, double yM[2]) const
    {
        if ( lambdaxy(yM) != 0 ){
            if ( j == 1 ){
                if ( k == 1 ){
                    return ( 16 * radius * radius * pi_ * yM[0] * yM[0] )  / pow( yM[0] * yM[0] + yM[1] * yM[1] , 3 ) - ( 4 * radius * radius * pi_ ) / pow( yM[0] * yM[0] + yM[1] * yM[1] , 2 ); 
                }
                else if ( k == 2 ){
                    return ( 16 * radius * radius * pi_ * yM[0] * yM[1] )  / pow( yM[0] * yM[0] + yM[1] * yM[1] , 3 );
                }
                else{
                    std::cout << "DDINVT_D1: k1: INVALID INDEX" << std::endl;
                    return 0;
                }
            }
            else if ( j == 2 ){
                if ( k == 1 ){
                    return ( 16 * radius * radius * pi_ * yM[0] * yM[1] )  / pow( yM[0] * yM[0] + yM[1] * yM[1] , 3 );
                }
                else if ( k == 2 ){
                    return ( 16 * radius * radius * pi_ * yM[1] * yM[1] )  / pow( yM[0] * yM[0] + yM[1] * yM[1] , 3 ) - ( 4 * radius * radius * pi_ ) / pow( yM[0] * yM[0] + yM[1] * yM[1] , 2 ); 
                } 
                else{
                    std::cout << "DDINVT_D1: k2: INVALID INDEX" << std::endl;
                    return 0;
                }
            }
            else{
                std::cout << "DDINVT_D1: ij: INVALID INDEX" << std::endl;
                return 0;
            }
        }
        else{
            return 0.0;
        }
    }

    //dim 2
    double DDInvt_d2(int i, int j, int k, int l, double yM[2]) const
    {
        if ( lambdaxy(yM) != 0 ){
            if( ( i == 1 && j == 1 ) || ( i == 2 && j == 2 ) ){
                if ( k == 1 ){
                    if ( l == 1 ){
                        return ( 8 * radius * radius * pi_ * yM[0] * yM[0] ) / ( 1 * pow( yM[0] * yM[0] + yM[1] * yM[1] , 4 ) ) - ( 4 * radius * radius * pi_ ) / ( 3 * pow( yM[0] * yM[0] + yM[1] * yM[1] , 3 ) );
                    }
                    else if ( l == 2 ){
                        return ( 8 * radius * radius * pi_ * yM[0] * yM[1] ) / ( 1 * pow( yM[0] * yM[0] + yM[1] * yM[1] , 4 ) );
                    }
                    else{
                        std::cout << "DDINVT_D2: l1: INVALID INDEX" << std::endl;
                        return 0;
                    }
                }
                else if ( k == 2 ){
                    if ( l == 1 ){
                        return ( 8 * radius * radius * pi_ * yM[0] * yM[1] ) / ( 1 * pow( yM[0] * yM[0] + yM[1] * yM[1] , 4 ) );
                    }
                    else if ( l == 2 ){
                        return ( 8 * radius * radius * pi_ * yM[1] * yM[1] ) / ( 1 * pow( yM[0] * yM[0] + yM[1] * yM[1] , 4 ) ) - ( 4 * radius * radius * pi_ ) / ( 3 * pow( yM[0] * yM[0] + yM[1] * yM[1] , 3 ) );
                    }
                    else{
                        std::cout << "DDINVT_D2: l2: INVALID INDEX" << std::endl;
                        return 0;
                    }
                }
                else{
                    std::cout << "DDINVT_D2: k1: INVALID INDEX" << std::endl;
                    return 0;
                }
            }
            else if( ( i == 1 && j == 2 ) || ( i == 2 && j == 1 ) ){
                if ( k == 1 ){
                    if ( l == 1 ){
                        return ( 24 * radius * radius * pi_ * yM[0] * yM[0] ) / ( 1 * pow( yM[0] * yM[0] + yM[1] * yM[1] , 4 ) ) - ( 4 * radius * radius * pi_ ) / ( 1 * pow( yM[0] * yM[0] + yM[1] * yM[1] , 3 ) );
                    }
                    else if ( l == 2 ){
                        return ( 24 * radius * radius * pi_ * yM[1] * yM[0] ) / ( 1 * pow( yM[0] * yM[0] + yM[1] * yM[1] , 4 ) );
                    }
                    else{
                        std::cout << "DDINVT_D2: l1: INVALID INDEX" << std::endl;
                        return 0;
                    }
                }
                else if ( k == 2 ){
                    if ( l == 1 ){
                        return ( 24 * radius * radius * pi_ * yM[1] * yM[0] ) / ( 1 * pow( yM[0] * yM[0] + yM[1] * yM[1] , 4 ) );
                    }
                    else if ( l == 2 ){
                        return ( 24 * radius * radius * pi_ * yM[1] * yM[1] ) / ( 1 * pow( yM[0] * yM[0] + yM[1] * yM[1] , 4 ) ) - ( 4 * radius * radius * pi_ ) / ( 1 * pow( yM[0] * yM[0] + yM[1] * yM[1] , 3 ) );
                    }
                    else{
                        std::cout << "DDINVT_D2: l2: INVALID INDEX" << std::endl;
                        return 0;
                    }
                }
                else{
                    std::cout << "DDINVT_D2: k2: INVALID INDEX" << std::endl;
                    return 0;
                }
            }
            else{
                std::cout << "DDINVT_D2: ij: INVALID INDEX" << std::endl;
                return 0;
            }
        }
        else{
            return 0.0;
        }
    }

    //Compute Delta Function
    double Delta(int i, int j) const
    {
        if ( i == j ){
            return 1.0;
        }
        else{
            return 0.0;
        }
    }

    //Compute phi,i
    double Dphi(int i, double yM[2]) const
    {
        return -yM[i - 1] * Invt_d1(i,yM);
    }

    //Compute psi,ijk
    double Dpsi(int i, int j, int k, double yM[2]) const
    {
        double D_out = 0;
        D_out = Delta(i,j) * ( -yM[k-1] * Invt_d1(k,yM) + radius * radius * yM[k-1] * ( Invt_d2(1,k,yM) + Invt_d2(2,k,yM) ) ); //I summation apply
        D_out = D_out - Delta(i,k) * ( yM[j-1] * Invt_d1(j,yM) - radius * radius * yM[j-1] * ( Invt_d2(1,j,yM) + Invt_d2(2,j,yM) ) ); //I summation apply
        D_out = D_out - yM[i-1] * ( Delta(j,k) * Invt_d1(j,yM) + yM[j-1] * DInvt_d1(j,k,yM) - radius * radius * ( Delta(j,k) * ( Invt_d2(1,j,yM) + Invt_d2(2,j,yM) ) + yM[j-1] * ( DInvt_d2(1,j,k,yM) + DInvt_d2(2,j,k,yM) ) ) );// I summation apply
        return D_out;
    }

    //Compute 8*pi*(1-nu)*S_ijkl(lambda)
    double coeff_S(int i,int j,int k,int l, double yM[2]) const
    {
        double Sout = 0;
        Sout = Delta(i,j) * Delta(k,l) * ( 2 * nu * Invt_d1(i,yM) - Invt_d1(k,yM) + radius * radius * ( Invt_d2(k,1,yM) + Invt_d2(k,2,yM) ) ); //I summation applies 
        Sout = Sout + ( Delta(i,k) * Delta(j,l) + Delta(j,k) * Delta(i,l) ) * ( radius * radius * ( Invt_d2(1,j,yM) + Invt_d2(2,j,yM) ) - Invt_d1(j,yM) + ( 1 - nu ) * ( Invt_d1(k,yM) + Invt_d1(l,yM) ) );
        return Sout;
    }

};
//------------------------------------------------//
//----------Define Dmat dolfin Expression---------//
class tensor_D : public Expression, public Eshelby
{
    public:

    double loc[2];

    tensor_D(double loc_in[2]) : Expression(3,3) //constructor w parameters
    { 
        loc[0] = loc_in[0]; 
        loc[1] = loc_in[1]; 
    } 

    //Compute Dmat
    double comp_D(int i, int j, int k, int l, double yM[2]) const 
    {
        double Dout = 0;
        Dout = coeff_S(i,j,k,l,yM) + 2 * nu * Delta(k,l) * yM[i-1] * DInvt_d1(i,j,yM);
        Dout = Dout + ( 1 - nu ) * ( Delta(i,l) * yM[k-1] * DInvt_d1(k,j,yM) + Delta(j,l) * yM[k-1] * DInvt_d1(k,i,yM) + Delta(i,k) * yM[l-1] * DInvt_d1(l,j,yM) + Delta(j,k) * yM[l-1] * DInvt_d1(l,i,yM) );
        Dout = Dout - Delta(i,j) * yM[k-1] * ( DInvt_d1(k,l,yM) - radius * radius * ( DInvt_d2(k,1,l,yM) + DInvt_d2(k,2,l,yM) ) ); //I summation apply
        Dout = Dout - ( Delta(i,k) * yM[j-1] + Delta(j,k) * yM[i - 1] ) * ( DInvt_d1(j,l,yM) - radius * radius * ( DInvt_d2(1,j,l,yM) + DInvt_d2(2,j,l,yM) ) ); //I summation apply
        Dout = Dout - ( Delta(i,l) * yM[j-1] + Delta(j,l) * yM[i - 1] ) * ( DInvt_d1(j,k,yM) - radius * radius * ( DInvt_d2(1,j,k,yM) + DInvt_d2(2,j,k,yM) ) ); //I summation apply
        Dout = Dout - yM[i-1] * yM[j-1] * ( DDInvt_d1(j,l,k,yM) - radius * radius * ( DDInvt_d2(1,j,l,k,yM) + DDInvt_d2(2,j,l,k,yM) ) ); //I summation apply
        Dout = Dout / ( 8 * pi_ * ( 1 - nu ) );
        return Dout;
    }

    void eval(Array<double>& values, const Array<double>& x) const
    {
        double yM[2] = {0, 0};
        yM[0] = x[0] - loc[0]; 
        yM[1] = x[1] - loc[1];

        values[0] = comp_D(1,1,1,1,yM);
        values[1] = comp_D(1,1,2,2,yM);
        values[2] = comp_D(1,1,1,2,yM);
        values[3] = comp_D(2,2,1,1,yM);
        values[4] = comp_D(2,2,2,2,yM);
        values[5] = comp_D(2,2,1,2,yM);
        values[6] = comp_D(1,2,1,1,yM);
        values[7] = comp_D(1,2,2,2,yM);
        values[8] = comp_D(1,2,1,2,yM);
    }
};
//------------------------------------------------//
//----------Define Hmat dolfin Expression---------//
class tensor_H : public Expression, public Eshelby
{
    public:

    double loc[2];

    tensor_H(double loc_in[2]) : Expression(3,3) //constructor w parameters //(3,3): for visualization only
    { 
        loc[0] = loc_in[0]; 
        loc[1] = loc_in[1]; 
    } 

    double comp_H(int i, int j, int k, double yM[2]) const 
    {   
        double Hout = 0;
        for (int m = 1; m < 3; m++){
            for (int n = 1; n < 3; n++){
                for (int l = 1; l < 3; l++){
                    Hout = Hout + Dpsi(m,n,i,yM) * Delta(j,m) * Delta(k,n);
                    Hout = Hout - 2 * nu * Delta(m,j) * Delta(m,k) * Dphi(i,yM);
                    Hout = Hout - 4 * ( 1 - nu ) * Delta(i,j) * Delta(k,l) * Dphi(l,yM);
                }
            }
        }
        Hout = Hout / ( 8 * pi_ * ( 1 - nu ) );
        return Hout;
    }

     void eval(Array<double>& values, const Array<double>& x) const
    {
        double yM[2] = {0, 0};
        yM[0] = x[0] - loc[0];
        yM[1] = x[1] - loc[1];

        values[0] = comp_H(1,1,1,yM);
        values[1] = comp_H(1,2,2,yM);
        values[2] = comp_H(1,1,2,yM);
        values[3] = comp_H(2,1,1,yM);
        values[4] = comp_H(2,2,2,yM);
        values[5] = comp_H(2,1,2,yM);

        //can only handle 4(2by2) or 9(3by3)
        values[6] = 0.0;
        values[7] = 0.0;
        values[8] = 0.0;
    }

};
//------------------------------------------------//
//----------Define STZ Helper Functions-----------//
class stz_helper
{   
    public:

    /*Compute initial stz position/strain energy using pesudo-number generator
    //Goal: Generate locs[][2]
    //Input: seed1(ij) seed2(kl) -> taken to be "12" "67"
    //Output: locs[][2] contains (x,y) coordinate of stzs */
    //--> add "flag": 'init' == 0 or 'add' == 1
    //--> add "N_new": current number of stzs

    void stz_init(int ij, int kl, int N0, std::vector<double>& locs, std::vector<double>& engs, int flag, int N_new ){

        RandomInitialise(ij, kl);

        if ( flag == 0 ){ //create locs and engs

            for (int i = 0; i < N0; i++){
                double uni_x = RandomUniform();
                double uni_y = RandomUniform();
                //Strain Energy //use box-muller method, obtain independent variables std_x, std_y
                double std_x = phi_c + phi_s * sqrt( -2.0 * log(uni_x) ) * cos( 2.0 * pi_ * uni_y );
                double std_y = phi_c + phi_s * sqrt( -2.0 * log(uni_x) ) * sin( 2.0 * pi_ * uni_y );
                //Position
                double stz_x = uni_x * L;
                double stz_y = uni_y * h;
                //Add exclusion zone free of STZ centers at the boundary
                //Add restriction on values within 1 standard derivation
                bool not_overlap_flag = false;
                while ( not_overlap_flag == false ){
                    uni_x = RandomUniform();
                    uni_y = RandomUniform();
                    stz_x = uni_x * L;
                    stz_y = uni_y * h;
                    not_overlap_flag = true;
                    for (int stz_in = 0; stz_in < i; stz_in ++ ){ //overlap check
                        auto stz_in_x = locs[0 + stz_in * 2];
                        auto stz_in_y = locs[1 + stz_in * 2];
                        auto dist = sqrt( pow( ( stz_x - stz_in_x ) , 2 ) + pow( ( stz_y - stz_in_y ) , 2 ) );
                        if ( (stz_x < 2 * radius) || (stz_y < 2 * radius) || ((L - stz_x) < 2 * radius) || ((h - stz_y) < 2 * radius) || dist < 2 * radius ){ not_overlap_flag = false; }
                    }
                }
                while ( ( abs(std_x - phi_c) > phi_threshold ) || ( abs(std_y - phi_c) > phi_threshold ) ){
                    uni_x = RandomUniform();
                    uni_y = RandomUniform();
                    std_x = phi_c + phi_s * sqrt( -2.0 * log(uni_x) ) * cos( 2.0 * pi_ * uni_y );
                    std_y = phi_c + phi_s * sqrt( -2.0 * log(uni_x) ) * sin( 2.0 * pi_ * uni_y );
                }

                locs[0 + i * 2] = stz_x; //"2" is for 2d hardcode
                locs[1 + i * 2] = stz_y;
                //Create a 1 or 0 random-number generator 
                const int randomBit = std::rand() % 2;
                if ( i < N0 / 2 ){
                //if ( randomBit == 0 ){
                    engs[2 * i] = std_x; //shear strain energy must be positive
                //}
                //else{
                    engs[2 * i + 1] = std_y;
                //}
                }
            }
        }
        else if ( flag == 1 ){ //update engs based on new locs

            for (int i = N0; i < N_new; i++){ //only update new generated stzs
                double uni_x = RandomUniform();
                double uni_y = RandomUniform();
                //Strain Energy //use box-muller method, obtain independent variables std_x, std_y
                double std_x = phi_c + phi_s * sqrt( -2.0 * log(uni_x) ) * cos( 2.0 * pi_ * uni_y );
                double std_y = phi_c + phi_s * sqrt( -2.0 * log(uni_x) ) * sin( 2.0 * pi_ * uni_y );
                //Add exclusion zone free of STZ centers at the boundary
                //Add restriction on values within 1 standard derivation
                while ( ( abs(std_x - phi_c) > phi_threshold ) || ( abs(std_y - phi_c) > phi_threshold ) ){
                    uni_x = RandomUniform();
                    uni_y = RandomUniform();
                    std_x = phi_c + phi_s * sqrt( -2.0 * log(uni_x) ) * cos( 2.0 * pi_ * uni_y );
                    std_y = phi_c + phi_s * sqrt( -2.0 * log(uni_x) ) * sin( 2.0 * pi_ * uni_y );
                }
                //Create a 1 or 0 random-number generator 
                const int randomBit = std::rand() % 2;
                if ( randomBit == 0 ){
                //if ( ( N_new - N0 ) / 2 % 0 ){
                    engs.push_back(std_x); //shear strain energy must be positive
                }
                else{
                    engs.push_back(std_y);
                //}
                }
            }
        
        }
        else if ( flag == 2 ){ //too many initial STZs, need to check overlapping
        }
        else{ std::cout<<"PLEASE PROVIDE CORRECT FLAG"<<std::endl; }

    }

    //Initialzation
    void RandomInitialise(int ij,int kl)
    {
        double s,t;
        int ii,i,j,k,l,jj,m;

        /*
            Handle the seed range errors
                First random number seed must be between 0 and 31328
                Second seed must have a value between 0 and 30081
        */
        if (ij < 0 || ij > 31328 || kl < 0 || kl > 30081) {
                ij = 1802;
                kl = 9373;
        }

        i = (ij / 177) % 177 + 2;
        j = (ij % 177)       + 2;
        k = (kl / 169) % 178 + 1;
        l = (kl % 169);

        for (ii=0; ii<97; ii++) {
            s = 0.0;
            t = 0.5;
            for (jj=0; jj<24; jj++) {
                m = (((i * j) % 179) * k) % 179;
                i = j;
                j = k;
                k = m;
                l = (53 * l + 1) % 169;
                if (((l * m % 64)) >= 32)
                    s += t;
                    t *= 0.5;
                }
            randU[ii] = s;
        }

        randC    = 362436.0 / 16777216.0;
        randCD   = 7654321.0 / 16777216.0;
        randCM   = 16777213.0 / 16777216.0;
        i97  = 97;
        j97  = 33;
        test = true;
    }

    //Uniform psedo-number generator
    double RandomUniform(void)
    {
        double uni;
        int seed1, seed2;

        uni = randU[i97-1] - randU[j97-1];
        if (uni <= 0.0)
            uni++;
        randU[i97-1] = uni;
        i97--;
        if (i97 == 0)
            i97 = 97;
        j97--;
        if (j97 == 0)
            j97 = 97;
        randC -= randCD;
        if (randC < 0.0)
            randC += randCM;
        uni -= randC;
        if (uni < 0.0)
            uni++;

        return(uni);
    }

    //C++ built-in random gaussian distribution
    void random_gaussian_distribution(std::vector<double>& eng, int flag, int N0, int N_new){
        
        std::default_random_engine generator;
        std::normal_distribution<double> distribution(phi_c,phi_s);

        double engs_gauss;

        if ( flag == 0 ){ //initial

            for (int i = 0; i < N0; i ++){
                engs_gauss = distribution(generator);
                while ( engs_gauss <= phi_c - phi_threshold || engs_gauss >= phi_c + phi_threshold ){ 
                    engs_gauss = distribution(generator);
                }
                eng[i] = engs_gauss;
            }
        
        }
        else if ( flag == 1 ){ //add new stzs

            for (int i = N0; i < N_new; i ++){
                engs_gauss = distribution(generator);
                while ( engs_gauss <= phi_c - phi_threshold || engs_gauss >= phi_c + phi_threshold ){
                    engs_gauss = distribution(generator);
                }
                eng[i] = engs_gauss;
            }

        }

    }

    //C++ built-in random uniform distribution
    void random_uniform_distribution(std::vector<double>& locs, int flag, int N0, int N_new){

        std::default_random_engine generator;
        std::uniform_real_distribution<double> distribution(0.0,1.0);

        double x_loc;
        double y_loc;

        if ( flag == 0 ){
            for (int i = 0; i < N0; i ++){
                double unit_x = distribution(generator);
                double unit_y = distribution(generator);
                x_loc = unit_x * L;
                y_loc = unit_y * h;
                bool not_overlap_flag = false;
                while ( not_overlap_flag == false ){
                    double unit_x = distribution(generator);
                    double unit_y = distribution(generator);
                    x_loc = unit_x * L;
                    y_loc = unit_y * h;
                    not_overlap_flag = true;
                    for (int stz_in = 0; stz_in < i; stz_in ++ ){ //overlap check
                        auto stz_in_x = locs[0 + stz_in * 2];
                        auto stz_in_y = locs[1 + stz_in * 2];
                        auto dist = sqrt( pow( ( x_loc - stz_in_x ) , 2 ) + pow( ( y_loc - stz_in_y ) , 2 ) );
                        if ( (x_loc < 2 * radius) || (y_loc < 2 * radius) || ((L - x_loc) < 2 * radius) || ((h - y_loc) < 2 * radius) || dist < 2 * radius ){ not_overlap_flag = false; }
                    }
                }
                locs[0 + i * 2] = x_loc;
                locs[1 + i * 2] = y_loc;
            }
        }
        /*
        else if ( flag == 1 ){
            for (int i = N0; i < N_new; i ++){
                double unit_x = distribution(generator);
                double unit_y = distribution(generator);
                x_loc = unit_x * L;
                y_loc = unit_y * h;
                while ((x_loc < 2 * radius) || (y_loc < 2 * radius) || ((L - x_loc) < 2 * radius) || ((h - y_loc) < 2 * radius)){
                    double unit_x = distribution(generator);
                    double unit_y = distribution(generator);
                    x_loc = unit_x * L;
                    y_loc = unit_y * h;
                }
                locs[0 + i * 2] = x_loc;
                locs[1 + i * 2] = y_loc;
            }
        }
        */

    }

    /*Compute shear strain energy (in-place)
    //Goal: phi_energy = 1 / ( 3  * G ) * sigma_e * sigma_e, sigma_e = sqrt( 3/2 * sigma_dev_ij * sigma_dev_ij )
    //Input: total stress, initialized phi_energy tensor
    //Output: shear strain energy (function that can be evaluated at any point in the domain)*/
    void strain_energy(std::shared_ptr<Function> sigma_total, std::shared_ptr<Function>& phi_energy)
    {
        std::vector<double> sts11_vec, sts12_vec, sts21_vec, sts22_vec;

        auto sts11_ptr = std::make_shared<Function>((*sigma_total)[0]);
        auto sts12_ptr = std::make_shared<Function>((*sigma_total)[1]);
        auto sts21_ptr = std::make_shared<Function>((*sigma_total)[2]);
        auto sts22_ptr = std::make_shared<Function>((*sigma_total)[3]);

        sts11_ptr->vector()->get_local(sts11_vec);
        sts12_ptr->vector()->get_local(sts12_vec);
        sts21_ptr->vector()->get_local(sts21_vec);
        sts22_ptr->vector()->get_local(sts22_vec);

        auto nnode = sts11_vec.size();

        //Compute tr(stress)
        std::vector<double> sts_tr_vec(nnode, 0.0);
        std::vector<double> sigma_zz(nnode, 0.0);

        for (int i = 0; i < sts11_vec.size(); i++){ //loop over number of ptrs
            sigma_zz[i] = nu * ( sts11_vec[i] + sts22_vec[i] );
            sts_tr_vec[i] = sts11_vec[i] + sts22_vec[i] + sigma_zz[i]; 
        }

        //Compute derivatoric stress tensor components
        std::vector<double> sts11_dev_vec(nnode, 0.0), sts12_dev_vec(nnode, 0.0), sts21_dev_vec(nnode, 0.0), sts22_dev_vec(nnode, 0.0);
        std::vector<double> sts33_dev_vec(nnode, 0.0);

        for (int j = 0; j < sts11_dev_vec.size(); j++){ //loop over number of ptrs
            sts11_dev_vec[j] = sts11_vec[j] - 1.0 / 3.0 * sts_tr_vec[j];
            sts12_dev_vec[j] = sts12_vec[j];
            sts21_dev_vec[j] = sts21_vec[j];
            sts22_dev_vec[j] = sts22_vec[j] - 1.0 / 3.0 * sts_tr_vec[j];
            sts33_dev_vec[j] = sigma_zz[j] -  1.0 / 3.0 * sts_tr_vec[j];
        }

        std::vector<double> sigma_e_vec(nnode, 0.0);
        std::vector<double> phi_energy_vec(nnode, 0.0); //sigma_e & strain energy 

        for (int i = 0; i < nnode; i++){
            phi_energy_vec[i] = 1.0 / ( 2.0 * G ) * ( pow( sts11_dev_vec[i], 2 ) + pow( sts12_dev_vec[i], 2 ) + pow( sts21_dev_vec[i], 2 ) + pow( sts22_dev_vec[i], 2 ) + pow(sts33_dev_vec[i], 2) ); //double 1.0/3.0  int 1/3 = 0
        }

        std::vector<double> phi_not_use_vec(nnode,0.0);

        auto phi_ptr = std::make_shared<Function>((*phi_energy)[0]); //no subsystem for finite element
        auto phi_not_use_ptr = std::make_shared<Function>((*phi_energy)[1]);

        phi_ptr->vector()->set_local(phi_energy_vec);
        phi_not_use_ptr->vector()->set_local(phi_not_use_vec);

        phi_ptr->vector()->apply("insert");
        phi_not_use_ptr->vector()->apply("insert");

        assign(phi_energy,{phi_ptr,phi_not_use_ptr});

    }

    /*Compute strain energy evaluated at one stz location, given coordinate of stz's center
    //Goal: phi_energy(stz i), locs : locations of stzs, locs_energy, number of stzs
    //Input: strain energy (function), locs (function)*/
    void strain_energy_local(std::shared_ptr<Function> phi_energy, std::vector<double> locs, std::vector<double>& locs_energy, int stz_iter)
    {
        Array<double> point_value(2);
        Array<double> point_locs(2);

        point_locs[0] = locs[0 + stz_iter * 2];
        point_locs[1] = locs[1 + stz_iter * 2];
        phi_energy->eval(point_value,point_locs); //interpolate
        locs_energy[stz_iter] = point_value[0];

        //std::cout<<"Rank: "<<MPI::rank(MPI_COMM_WORLD)<<" "<<"ptr num: "<<stz_iter<<" "<<"eval: "<<point_value[0]<<std::endl;

    }

    /*Compute total stress tensor (in-place)
    //Goal: total stress = image field stress + tilde field stress
    //Input: image field stress, tilde field stress, total stress
    //Output: modified total stress in place*/
    void total_stress(std::shared_ptr<Function> sigma_image, std::shared_ptr<Function> sigma_tilde, std::shared_ptr<Function>& sigma_total)
    {
        sigma_total->vector()->zero(); //Initialization step1
        sigma_total->interpolate(*sigma_image); //step2
        //sigma_total->vector()->axpy(1.0, *sigma_image->vector()); //sigma_total = sigma_total + 1.0 * sigma_image
        sigma_total->vector()->axpy(1.0, *sigma_tilde->vector()); //sigma_total = sigma_total + 1.0 * sigma_tilde
    }

    /*Compute total strain tensor (in-place)
    //Goal: total strain = image field strain + tilde field strain
    //Input: image field strain, tilde field strain, total strain
    //Output: modified total strain in place*/
    void total_strain(std::shared_ptr<Function> eps_image, std::shared_ptr<Function> eps_tilde, std::shared_ptr<Function>& eps_total)
    {
        eps_total->vector()->zero(); //Initialization step 1
        eps_total->interpolate(*eps_image); //step 2
        //eps_total->vector()->axpy(1.0, *eps_image->vector()); //sigma_total = sigma_total + 1.0 * sigma_image
        eps_total->vector()->axpy(1.0, *eps_tilde->vector()); //sigma_total = sigma_total + 1.0 * sigma_tilde
    }

    /*Compute total displs vector (in-place)
    //Goal: total displs = image field displs + tilde field displs
    //Input: image field displs, tilde field displs, total displs
    //Output: modified total displs in place*/
    void total_displs(std::shared_ptr<Function> displs_image, std::shared_ptr<Function> displs_tilde, std::shared_ptr<Function>& displs_total)
    {   
        //displs_image: w periodic, displs_tilde, displs_total: w/o periodic
        //1. initialize displs_tilde, 2. interpolate displs_image on nonperiodic, store in displs_total, 3. add displs_tilde to displs_total
        //stress, strain should follow the same logic
        
        displs_total->vector()->zero(); //Initialization step1
        displs_total->interpolate(*displs_image); // step2
        //displs_total->vector()->axpy(1.0, *displs_image->vector()); //sigma_total = sigma_total + 1.0 * sigma_image
        displs_total->vector()->axpy(1.0, *displs_tilde->vector()); //sigma_total = sigma_total + 1.0 * sigma_tilde //step3
    }

    /*Compute image stress tensor (in-place)
    //Goal: image field stress = Lijkl * image field strain
    //Input: image field strain, image field stress 
    //Output: modified image stress in place*/
    void image_stress(std::shared_ptr<Function> eps_image, std::shared_ptr<Function>& sigma_image)
    {
        //Initialize strain vectors 
        std::vector<double> eps11_vec, eps12_vec, eps21_vec, eps22_vec; //for storing strain tensor components

        //Get strain components
        auto eps11_ptr = std::make_shared<Function>((*eps_image)[0]);
        auto eps12_ptr = std::make_shared<Function>((*eps_image)[1]);
        auto eps21_ptr = std::make_shared<Function>((*eps_image)[2]);
        auto eps22_ptr = std::make_shared<Function>((*eps_image)[3]);

        //Obtain strain component vectors
        eps11_ptr->vector()->get_local(eps11_vec);
        eps12_ptr->vector()->get_local(eps12_vec);
        eps21_ptr->vector()->get_local(eps21_vec);
        eps22_ptr->vector()->get_local(eps22_vec);

        //Get vector size 
        auto nnode = eps11_vec.size();

        std::vector<double> eps_tr_vec(nnode,0.0); //for stroing tr(strain)
        //Initialize stress component vector w same size as strain component vector
        //Set initial value as 0
        std::vector<double> sts11_vec(nnode,0.0), sts12_vec(nnode,0.0), sts21_vec(nnode,0.0), sts22_vec(nnode,0.0); //for storing strain tensor components

        //Get stress components
        auto sts11_ptr = std::make_shared<Function>((*sigma_image)[0]);
        auto sts12_ptr = std::make_shared<Function>((*sigma_image)[1]);
        auto sts21_ptr = std::make_shared<Function>((*sigma_image)[2]);
        auto sts22_ptr = std::make_shared<Function>((*sigma_image)[3]);

        //Compute tr(strain)
        for (int i = 0; i < eps11_vec.size(); i++){ //loop over number of ptrs
            eps_tr_vec[i] = eps11_vec[i] + eps22_vec[i]; 
        }

        //Compute stress component vectors
        for (int j = 0; j < eps11_vec.size(); j++){ //loop over number of ptrs
            sts11_vec[j] = E / ( 1 + nu ) * ( eps11_vec[j] + nu / ( 1 - 2 * nu ) * eps_tr_vec[j] );
            sts22_vec[j] = E / ( 1 + nu ) * ( eps22_vec[j] + nu / ( 1 - 2 * nu ) * eps_tr_vec[j] );
            sts12_vec[j] = E / ( 1 + nu ) * ( eps12_vec[j]                                       );
            sts21_vec[j] = E / ( 1 + nu ) * ( eps21_vec[j]                                       );
        }

        //Set stress components back into stress function
        sts11_ptr->vector()->set_local(sts11_vec);
        sts12_ptr->vector()->set_local(sts12_vec);
        sts21_ptr->vector()->set_local(sts21_vec);
        sts22_ptr->vector()->set_local(sts22_vec);
        
        sts11_ptr->vector()->apply("insert");
        sts12_ptr->vector()->apply("insert");
        sts21_ptr->vector()->apply("insert");
        sts22_ptr->vector()->apply("insert");

        assign(sigma_image,{sts11_ptr,sts12_ptr,sts21_ptr,sts22_ptr});
    
    }

    /*image_strain <- image_displs is computed through solving weak form (or projection) ( See also project.ufl ) */

    /*image_displs is computed through solving weak form ( See also vf.ufl ) */

    /*Compute tilde stress tensor (in-place)
    //Goal: tilde field stress rate = Lijkl * tilde field strain rate
    //Input: tilde field strain, tilde field stress 
    //Output: modified tilde stress in place*/
    void tilde_stress(std::shared_ptr<Function> eps_tilde, std::shared_ptr<Function>& sigma_tilde)
    {
        //Initialize strain vectors 
        std::vector<double> eps11_vec, eps12_vec, eps21_vec, eps22_vec; //for storing strain tensor components

        //Get strain components
        auto eps11_ptr = std::make_shared<Function>((*eps_tilde)[0]);
        auto eps12_ptr = std::make_shared<Function>((*eps_tilde)[1]);
        auto eps21_ptr = std::make_shared<Function>((*eps_tilde)[2]);
        auto eps22_ptr = std::make_shared<Function>((*eps_tilde)[3]);

        //Obtain strain component vectors
        eps11_ptr->vector()->get_local(eps11_vec);
        eps12_ptr->vector()->get_local(eps12_vec);
        eps21_ptr->vector()->get_local(eps21_vec);
        eps22_ptr->vector()->get_local(eps22_vec);

        //Get vector size 
        auto nnode = eps11_vec.size();

        std::vector<double> eps_tr_vec(nnode,0.0); //for stroing tr(strain)
        //Initialize stress component vector w same size as strain component vector
        //Set initial value as 0
        std::vector<double> sts11_vec(nnode,0.0), sts12_vec(nnode,0.0), sts21_vec(nnode,0.0), sts22_vec(nnode,0.0); //for storing strain tensor components

        //Get stress components
        auto sts11_ptr = std::make_shared<Function>((*sigma_tilde)[0]);
        auto sts12_ptr = std::make_shared<Function>((*sigma_tilde)[1]);
        auto sts21_ptr = std::make_shared<Function>((*sigma_tilde)[2]);
        auto sts22_ptr = std::make_shared<Function>((*sigma_tilde)[3]);

        //Compute tr(strain)
        for (int i = 0; i < eps11_vec.size(); i++){ //loop over number of ptrs
            eps_tr_vec[i] = eps11_vec[i] + eps22_vec[i]; 
        }

        //Compute stress component vectors
        for (int j = 0; j < eps11_vec.size(); j++){ //loop over number of ptrs
            sts11_vec[j] = E / ( 1 + nu ) * ( eps11_vec[j] + nu / ( 1 - 2 * nu ) * eps_tr_vec[j] );
            sts22_vec[j] = E / ( 1 + nu ) * ( eps22_vec[j] + nu / ( 1 - 2 * nu ) * eps_tr_vec[j] );
            sts12_vec[j] = E / ( 1 + nu ) * ( eps12_vec[j]                                       );
            sts21_vec[j] = E / ( 1 + nu ) * ( eps21_vec[j]                                       );
        }

        //Set stress components back into stress function
        sts11_ptr->vector()->set_local(sts11_vec);
        sts12_ptr->vector()->set_local(sts12_vec);
        sts21_ptr->vector()->set_local(sts21_vec);
        sts22_ptr->vector()->set_local(sts22_vec);
        
        sts11_ptr->vector()->apply("insert");
        sts12_ptr->vector()->apply("insert");
        sts21_ptr->vector()->apply("insert");
        sts22_ptr->vector()->apply("insert");

        assign(sigma_tilde,{sts11_ptr,sts12_ptr,sts21_ptr,sts22_ptr});
    }   

    /*Compute tilde strain tensor (in-place) (UNVERIFIED)
    //Goal: tilde field strain rate = Dijkl * eigenstrain rate
    //Input: Dmat, eigenstrain, tilde field strain 
    //Output: modified tilde strain in place*/
    void tilde_strain(std::shared_ptr<Function> Dmat, std::vector<double> eps_eigen, std::shared_ptr<Function>& eps_tilde, int stz_iter)
    {
        //Initialize Dmat component vectors 
        //Note: Dmat is constructed in vogit notation, however, eps_eigen_rate, eps_tilde_rate are NOT
        //Dmat possess major symmetry, but not minor symmetry
        //In this notation, the symmetry of strain tensor is considered
        std::vector<double> D1111_vec, D1122_vec, D1112_vec, D2211_vec, D2222_vec, D2212_vec, D1211_vec, D1222_vec, D1212_vec; //for storing Dmat tensor components

        //Get Dmat components
        auto D1111_ptr = std::make_shared<Function>((*Dmat)[0]);
        auto D1122_ptr = std::make_shared<Function>((*Dmat)[1]);
        auto D1112_ptr = std::make_shared<Function>((*Dmat)[2]);
        auto D2211_ptr = std::make_shared<Function>((*Dmat)[3]);
        auto D2222_ptr = std::make_shared<Function>((*Dmat)[4]);
        auto D2212_ptr = std::make_shared<Function>((*Dmat)[5]);
        auto D1211_ptr = std::make_shared<Function>((*Dmat)[6]);
        auto D1222_ptr = std::make_shared<Function>((*Dmat)[7]);
        auto D1212_ptr = std::make_shared<Function>((*Dmat)[8]);

        //Obtain Dmat component vectors
        D1111_ptr->vector()->get_local(D1111_vec);
        D1122_ptr->vector()->get_local(D1122_vec);
        D1112_ptr->vector()->get_local(D1112_vec);
        D2211_ptr->vector()->get_local(D2211_vec);
        D2222_ptr->vector()->get_local(D2222_vec);
        D2212_ptr->vector()->get_local(D2212_vec);
        D1211_ptr->vector()->get_local(D1211_vec);
        D1222_ptr->vector()->get_local(D1222_vec);
        D1212_ptr->vector()->get_local(D1212_vec);

        //Get eigenstrain components
        auto eigeneps11_iter = eps_eigen[0 + stz_iter * 4];
        auto eigeneps12_iter = eps_eigen[1 + stz_iter * 4];
        auto eigeneps21_iter = eps_eigen[2 + stz_iter * 4];
        auto eigeneps22_iter = eps_eigen[3 + stz_iter * 4];

        //Get size of vector
        auto nnode = D1111_vec.size();

        //Initialize tilde strain components vector
        std::vector<double> eps11_vec(nnode, 0.0), eps12_vec(nnode, 0.0), eps21_vec(nnode, 0.0), eps22_vec(nnode, 0.0);

        //Perform Dmat_ijkl eigeneps_kl contraction
        for (int i = 0; i < nnode; i++){
            eps11_vec[i] = D1111_vec[i] * eigeneps11_iter + D1112_vec[i] * eigeneps12_iter + D1112_vec[i] * eigeneps21_iter + D1122_vec[i] * eigeneps22_iter;
            eps12_vec[i] = D1211_vec[i] * eigeneps11_iter + D1212_vec[i] * eigeneps12_iter + D1212_vec[i] * eigeneps21_iter + D1222_vec[i] * eigeneps22_iter;
            eps21_vec[i] = D1211_vec[i] * eigeneps11_iter + D1212_vec[i] * eigeneps12_iter + D1212_vec[i] * eigeneps21_iter + D1222_vec[i] * eigeneps22_iter;
            eps22_vec[i] = D2211_vec[i] * eigeneps11_iter + D2212_vec[i] * eigeneps12_iter + D2212_vec[i] * eigeneps21_iter + D2222_vec[i] * eigeneps22_iter;
        }

        //Get tilde strain component pointer 
        auto eps11_ptr = std::make_shared<Function>((*eps_tilde)[0]);
        auto eps12_ptr = std::make_shared<Function>((*eps_tilde)[1]);
        auto eps21_ptr = std::make_shared<Function>((*eps_tilde)[2]);
        auto eps22_ptr = std::make_shared<Function>((*eps_tilde)[3]);

        //Set strain component vectors back to function
        eps11_ptr->vector()->set_local(eps11_vec);
        eps12_ptr->vector()->set_local(eps12_vec);
        eps21_ptr->vector()->set_local(eps21_vec);
        eps22_ptr->vector()->set_local(eps22_vec);

        eps11_ptr->vector()->apply("insert");
        eps12_ptr->vector()->apply("insert");
        eps21_ptr->vector()->apply("insert");
        eps22_ptr->vector()->apply("insert");

        assign(eps_tilde,{eps11_ptr,eps12_ptr,eps21_ptr,eps22_ptr});

    }

    /*Compute tilde displacement vector (in-place) (UNVERIFIED)
    //Goal: tilde displacement rate = Hijk * eigenstrain rate
    //Input: Hijk, eigenstrain rate, tilde displacement rate 
    //Output: tilde displacement rate (in-place) */
    void tilde_displs(std::shared_ptr<Function> Hmat, std::vector<double> eps_eigen, std::shared_ptr<Function>& u_tilde, int stz_iter)
    {   
        //
        std::vector<double> H111_vec, H122_vec, H112_vec, H211_vec, H222_vec, H212_vec;

        auto H111_ptr = std::make_shared<Function>((*Hmat)[0]);
        auto H122_ptr = std::make_shared<Function>((*Hmat)[1]);
        auto H112_ptr = std::make_shared<Function>((*Hmat)[2]);
        auto H211_ptr = std::make_shared<Function>((*Hmat)[3]);
        auto H222_ptr = std::make_shared<Function>((*Hmat)[4]);
        auto H212_ptr = std::make_shared<Function>((*Hmat)[5]);

        H111_ptr->vector()->get_local(H111_vec);
        H122_ptr->vector()->get_local(H122_vec);
        H112_ptr->vector()->get_local(H112_vec);
        H211_ptr->vector()->get_local(H211_vec);
        H222_ptr->vector()->get_local(H222_vec);
        H212_ptr->vector()->get_local(H212_vec);

        //Get eigenstrain components
        auto eigeneps11_iter = eps_eigen[0 + stz_iter * 4];
        auto eigeneps12_iter = eps_eigen[1 + stz_iter * 4];
        auto eigeneps21_iter = eps_eigen[2 + stz_iter * 4];
        auto eigeneps22_iter = eps_eigen[3 + stz_iter * 4];

        //Get size of vector
        auto nnode = H111_vec.size();

        std::vector<double> u1_vec(nnode, 0.0), u2_vec(nnode, 0.0);

        for (int i = 0; i < nnode; i++){
            u1_vec[i] = H111_vec[i] * eigeneps11_iter + H112_vec[i] * eigeneps12_iter + H112_vec[i] * eigeneps21_iter + H122_vec[i] * eigeneps22_iter;
            u2_vec[i] = H211_vec[i] * eigeneps11_iter + H212_vec[i] * eigeneps12_iter + H212_vec[i] * eigeneps21_iter + H222_vec[i] * eigeneps22_iter;
        }

        auto u1_ptr = std::make_shared<Function>((*u_tilde)[0]);
        auto u2_ptr = std::make_shared<Function>((*u_tilde)[1]);

        u1_ptr->vector()->set_local(u1_vec);
        u2_ptr->vector()->set_local(u2_vec);

        u1_ptr->vector()->apply("insert");
        u2_ptr->vector()->apply("insert");

        assign(u_tilde,{u1_ptr,u2_ptr});

    }

    /*Compute eigen strain rate tensor (in-place) 
    //Goal: eigenstrain rate = shear strain rate / 2 * ( s_i n_j + s_j n_i )
    //Input: shear strain rate (scalar function), s_i n_i (vector function), eigenstrain (tensor function)
    //Output: eigenstrain (tensor function) */
    void eigen_strain(std::vector<double>  shear_strain, std::vector<double> s_vec, std::vector<double> n_vec, std::vector<double>& eps_eigen, int stz_iter )
    {
        
        //shear strain now becomes incremental (step value)

        //
        auto ssr = shear_strain[stz_iter]; //gamma_step_vec

        auto s_1_vec = s_vec[0 + stz_iter * 2];
        auto s_2_vec = s_vec[1 + stz_iter * 2];
        auto n_1_vec = n_vec[0 + stz_iter * 2];
        auto n_2_vec = n_vec[1 + stz_iter * 2];

        //
        eps_eigen[0 + stz_iter * 4] = ssr / 2 * ( s_1_vec * n_1_vec + n_1_vec * s_1_vec );
        eps_eigen[1 + stz_iter * 4] = ssr / 2 * ( s_1_vec * n_2_vec + n_1_vec * s_2_vec );
        eps_eigen[2 + stz_iter * 4] = ssr / 2 * ( s_2_vec * n_1_vec + n_2_vec * s_1_vec );
        eps_eigen[3 + stz_iter * 4] = ssr / 2 * ( s_2_vec * n_2_vec + n_2_vec * s_2_vec );

    }

    /*Compute eigen strain rate tensor (volumertic part)*/
    //Created on Aug 24
    void eigen_strain_volumetric(std::vector<double>  tension_strain, std::vector<double> s_vec, std::vector<double> n_vec, std::vector<double>& eps_eigen_tension, int stz_iter )
    {
        
        //tension strain == epsilon_step_vec (as output in "biaxial_strain_rate_local" function) 
        //see 2D Tensor Transformation in writing notes

        //1. Need to create "eps_eigen_v", "eps_eigen_mpiv" vector

        //
        auto vs = tension_strain[stz_iter]; //gamma_step_vec

        //
        eps_eigen[0 + stz_iter * 4] = vs;
        eps_eigen[1 + stz_iter * 4] = 0;
        eps_eigen[2 + stz_iter * 4] = 0;
        eps_eigen[3 + stz_iter * 4] = vs;

    }

    /*Compute shear strain rate scalar (in-place)
    //Goal: shear strain rate scalar gamma* = 1 / B * 4 * ( 1 - nu ^ 2 ) / E * config_force Zsn (only shear components)
    //Input: shear strain rate (scalar function), configuration force (scalar function)
    //Output: shear strain rate (scalar function) */
    //--delete used "step_eigenstrain" input
    void shear_strain_rate_local(std::vector<double> config_force_Zsn, std::vector<double> gamma_rate, int stz_iter, std::vector<double>& gamma_vec, std::vector<int>& active_status, std::vector<double>& gamma_step_vec, int option, double dt_in, std::vector<int>& reactivate_indicator)
    {

        //active_status will record whether stz reaches threshold or not
        //gamma_step_vec save incremental change //replace previous at each step
        //add option: 1: instantanenous growth of eigenstrain 2: resolve eigenstrain growth history
        //add dt_in: refine dt when computing stz (double)
        //add reactive_indicator for reactivation (set INT_MAX once it reaches maximum strain)

        //2021-paper
        //gamma_rate[stz_iter] = 1 / B_ * 4 * ( 1 - nu * nu ) / E * config_force_Zsn[stz_iter];

        //JMPS
        gamma_rate[stz_iter] = 1 / ( B_ * pi_ * radius * radius * ( E / ( 4 * ( 1 - nu * nu ) ) ) ) * config_force_Zsn[stz_iter];

        if (gamma_rate[stz_iter] <= 0){
            std::cout<<"gamma_rate can not be less than zero : shear_strain_rate_local"<<std::endl;
        }

        if ( option == 1 ){
            active_status[stz_iter] = 2;
            if ( gamma_vec[stz_iter] + gamma_threshold_per_iter < gamma_threshold ){
                gamma_step_vec[stz_iter] = gamma_threshold_per_iter; //every step has to be this value
                gamma_vec[stz_iter]     += gamma_threshold_per_iter;
            }
            else{
                gamma_step_vec[stz_iter] = gamma_threshold - gamma_threshold_per_iter ; //only count the portion from last to threshold
                gamma_vec[stz_iter]      = gamma_threshold;
                reactivate_indicator[stz_iter] = INT_MAX;
            }
        }
        //Given gamma expression gamma_vec stores [time]!!!
        else if ( option == 2 ){
            auto gamma_val_stz = gamma_vec[stz_iter];

            if ( gamma_val_stz < t_terminate ){
        
                if ( gamma_val_stz + dt_in < t_terminate ){ //if it is below threshold

                    gamma_step_vec[stz_iter] = expression_eigenstrain(gamma_vec[stz_iter] + dt_in) - expression_eigenstrain(gamma_vec[stz_iter]);
                    gamma_vec[stz_iter] += dt_in;

                }
                else{
                    
                    active_status[stz_iter] = 2;
                    gamma_step_vec[stz_iter] = abs(expression_eigenstrain(t_terminate) - expression_eigenstrain(gamma_vec[stz_iter]));
                    gamma_vec[stz_iter]      = 0; //reset for next possible reactivation
                    //reactivate_indicator[stz_iter] = INT_MAX;
                
                }
            }
            else{

                active_status[stz_iter] = 2;
                gamma_step_vec[stz_iter] = 0;
                gamma_vec[stz_iter]      = 0;

            }

        }
        else if ( option == 3 ){
            //Check whether the gamma exceed threshold, set gamma_rate to 0.0 if it meets
            auto gamma_val_stz = gamma_vec[stz_iter];
        
            if ( gamma_val_stz + dt_in * gamma_rate[stz_iter] < gamma_threshold ){ //if it is below threshold

                if ( gamma_val_stz + dt_in * gamma_rate[stz_iter] < gamma_threshold_per_iter ){

                    gamma_step_vec[stz_iter]  = dt_in * gamma_rate[stz_iter];
                    gamma_vec[stz_iter]      += dt_in * gamma_rate[stz_iter];

                }
                else{

                    active_status[stz_iter] = 2;

                    if (gamma_val_stz > gamma_threshold_per_iter){

                        gamma_step_vec[stz_iter]  = 0;

                    }
                    else{

                        gamma_step_vec[stz_iter]  = gamma_threshold_per_iter - gamma_val_stz;

                    }

                    gamma_vec[stz_iter]  = 0;

                }

            }
            //else{
                
            //    active_status[stz_iter] = 2;
            //    gamma_step_vec[stz_iter] = gamma_threshold - dt_in * gamma_rate[stz_iter];
            //    gamma_vec[stz_iter]      = gamma_threshold; //update gamma-vec w stz is 1
            //    reactivate_indicator[stz_iter] = INT_MAX;
            
            //}

        }
        else{
            std::cout<<"WRONG option index: 1 , 2 : shear_strain_rate_local"<<std::endl;
        }

    }

    /*Compute biaxial tension strain rate local*/
    //Created on Aug 24
    //--delete used "step_eigenstrain" input
    void biaxial_strain_rate_local(std::vector<double> config_force_Zbeta, std::vector<double> epsilon_rate, int stz_iter, std::vector<double>& epsilon_vec, std::vector<int>& active_status, std::vector<double>& epsilon_step_vec, int option, double dt_in, std::vector<int>& reactivate_indicator)
    {   
        //JMPS
        epsilon_rate[stz_iter] = 1 / ( B_ * pi_ * radius * radius * ( E / ( 1 - nu * nu )  )   ) * config_force_Zbeta[stz_iter];

        //1. Need to create new "epsilon_rate", "epsilon_vec", "epsilon_step_vec"
        //2. Need to specify new "epsilon_threshold", "epsilon_threshold_per_iter"
        auto epsilon_val_stz = epsilon_vec[stz_iter];
        
        if ( epsilon_val_stz + dt_in * epsilon_rate[stz_iter] < epsilon_threshold ){ //if it is below threshold

            if ( epsilon_val_stz + dt_in * epsilon_rate[stz_iter] < epsilon_threshold_per_iter ){

                epsilon_step_vec[stz_iter]  = dt_in * epsilon_rate[stz_iter];
                epsilon_vec[stz_iter]      += dt_in * epsilon_rate[stz_iter];

            }
            else{

                active_status[stz_iter] = 2;

                if (epsilon_val_stz > epsilon_threshold_per_iter){

                    epsilon_step_vec[stz_iter]  = 0;

                }
                else{

                    epsilon_step_vec[stz_iter]  = epsilon_threshold_per_iter - epsilon_val_stz;

                }

                epsilon_vec[stz_iter]  = 0;

            }

        }


    }

    /*Compute configurational force scalar (in-place)
    //Goal: config_force_Zsn = image_stress_sn + sum( stress_tilde(site i) )
    //Input: image stress field, sum of local stz stress field, local vector s, local vector n
    //Output: config_force_Zsn in local coordinate
    */
    void config_force_local(std::shared_ptr<Function> sigma_image, std::shared_ptr<Function> sigma_stz_sum, std::vector<double>  s_vec, std::vector<double> n_vec, std::vector<double>& config_force_Zsn, std::vector<double> locs, int stz_iter)
    {   
        //Get image field stress tensor at this stz site
        Array<double> sts_image(4); //stress tensor
        Array<double> point_locs_image(2);  //nodal coordinate

        //Get nodal coordinate of stz
        point_locs_image[0] = locs[0 + stz_iter * 2];
        point_locs_image[1] = locs[1 + stz_iter * 2];
        
        sigma_image->eval(sts_image,point_locs_image);

        //Get tilde field stress tensor at this stz site
        Array<double> sts_stz_sum(4); //stress tensor

        sigma_stz_sum->eval(sts_stz_sum,point_locs_image);

        //std::cout<<"Rank: "<<MPI::rank(MPI_COMM_WORLD)<<" "<<"ptr num: "<<stz_iter<<" "<<"eval: "<<sts_image[0]<<std::endl;

        //Perform coordinate transformation on image field stress & tilde field stress (evaluated at current stz)
        //Here is summation !
        double sts12_vec_sn = 0.0;
        double sts_stz_sum12_vec_sn = 0.0;

        sts12_vec_sn             += s_vec[0 + stz_iter * 2] * sts_image[0] * n_vec[0 + stz_iter * 2]; //2 is hardcode for dim 2
        sts12_vec_sn             += s_vec[0 + stz_iter * 2] * sts_image[1] * n_vec[1 + stz_iter * 2];
        sts12_vec_sn             += s_vec[1 + stz_iter * 2] * sts_image[2] * n_vec[0 + stz_iter * 2];
        sts12_vec_sn             += s_vec[1 + stz_iter * 2] * sts_image[3] * n_vec[1 + stz_iter * 2];  

        sts_stz_sum12_vec_sn     += s_vec[0 + stz_iter * 2] * sts_stz_sum[0] * n_vec[0 + stz_iter * 2];
        sts_stz_sum12_vec_sn     += s_vec[0 + stz_iter * 2] * sts_stz_sum[1] * n_vec[1 + stz_iter * 2];
        sts_stz_sum12_vec_sn     += s_vec[1 + stz_iter * 2] * sts_stz_sum[2] * n_vec[0 + stz_iter * 2];
        sts_stz_sum12_vec_sn     += s_vec[1 + stz_iter * 2] * sts_stz_sum[3] * n_vec[1 + stz_iter * 2];

        auto cfZsn               = sts12_vec_sn + sts_stz_sum12_vec_sn;

        //Assign to config force vector
        config_force_Zsn[stz_iter] = cfZsn;

    }

    void config_force_local_test(std::shared_ptr<Function> sigma_total, std::vector<double>  s_vec, std::vector<double> n_vec, std::vector<double>& config_force_Zsn, std::vector<double> locs, int stz_iter){

        //Get image field stress tensor at this stz site
        Array<double> sts_total_val(4); //stress tensor
        Array<double> point_locs_total(2);  //nodal coordinate

        //Get nodal coordinate of stz
        point_locs_total[0] = locs[0 + stz_iter * 2];
        point_locs_total[1] = locs[1 + stz_iter * 2];

        sigma_total->eval(sts_total_val,point_locs_total);

        double cfZsn = 0.0;

        cfZsn += s_vec[0 + stz_iter * 2] * sts_total_val[0] * n_vec[0 + stz_iter * 2];
        cfZsn += s_vec[0 + stz_iter * 2] * sts_total_val[1] * n_vec[1 + stz_iter * 2];
        cfZsn += s_vec[1 + stz_iter * 2] * sts_total_val[2] * n_vec[0 + stz_iter * 2];
        cfZsn += s_vec[1 + stz_iter * 2] * sts_total_val[3] * n_vec[1 + stz_iter * 2];  

        config_force_Zsn[stz_iter] = cfZsn;

    }


    /*Compute configurational force for biaxial tension case*/
    //Created on Aug 24
    void config_force_local_biaxial(std::shared_ptr<Function> sigma_total, std::vector<double>  s_vec, std::vector<double> n_vec, std::vector<double>& config_force_Zbeta, std::vector<double> locs, int stz_iter){
        
        //1. Need new config_force_Zbeta

        //Get image field stress tensor at this stz site
        Array<double> sts_total_val(4); //stress tensor
        Array<double> point_locs_total(2);  //nodal coordinate

        //Get nodal coordinate of stz
        point_locs_total[0] = locs[0 + stz_iter * 2];
        point_locs_total[1] = locs[1 + stz_iter * 2];

        sigma_total->eval(sts_total_val,point_locs_total);

        //Z_beta_beta = Znn + Zss
        //Znn
        double cfZnn = 0.0;

        cfZnn += n_vec[0 + stz_iter * 2] * sts_total_val[0] * n_vec[1 + stz_iter * 2];
        cfZnn += n_vec[0 + stz_iter * 2] * sts_total_val[1] * n_vec[1 + stz_iter * 2];
        cfZnn += n_vec[1 + stz_iter * 2] * sts_total_val[2] * n_vec[0 + stz_iter * 2];
        cfZnn += n_vec[1 + stz_iter * 2] * sts_total_val[3] * n_vec[1 + stz_iter * 2];  

        //Zss
        double cfZss = 0.0;

        cfZss += n_vec[0 + stz_iter * 2] * sts_total_val[0] * n_vec[1 + stz_iter * 2];
        cfZss += n_vec[0 + stz_iter * 2] * sts_total_val[1] * n_vec[1 + stz_iter * 2];
        cfZss += n_vec[1 + stz_iter * 2] * sts_total_val[2] * n_vec[0 + stz_iter * 2];
        cfZss += n_vec[1 + stz_iter * 2] * sts_total_val[3] * n_vec[1 + stz_iter * 2];  

        config_force_Zbeta[stz_iter] = cfZnn + cfZss;

    }

    /*Compute summation of local stz stress field (in-place)
    //Goal: sum of local stz stress field = sum( local stz stress field ), !!!with a for loop outside!!!
    //Input: std::vector of function objects with the number equals the number of stz sites
    //Output: summation of local stz stress field as a function object 
    //NOTE: a cutoff radius may apply in the for loop if the summation of whole system takes too much time */
    void stz_stress_sum(std::shared_ptr<Function> sigma_stz_iter, std::shared_ptr<Function>& sigma_stz_sum)
    {
        sigma_stz_sum->vector()->axpy(1.0, *sigma_stz_iter->vector()); //sigma_stz_sum = sigma_stz_sum + 1.0 * sigma_stz_iter
    }

    /*Compute local coordinate (s,n) based on maximum shear direction (in-place)
    //Goal: 2 theta_max = arctan( ( sigma_yy - sigma_xx ) / ( 2 * sigma_xy ) ) 
    //s = ( cos(theta_max), sin(theta_max) ); n = ( (-sin(theta_max), cos(theta_max) ) ) 
    //Input: total stress field tensor, s, n (pre-defined vector), stz_iter: current stz_number
    //Output: s,n local coordinate vector !only evaluate at current activated stz site!*/
    void coord_sn_local(std::shared_ptr<Function> sigma_total, std::vector<double>& s_vec, std::vector<double>& n_vec, std::vector<double> locs, int stz_iter)
    {   
        //
        Array<double> point_value(4); //stress tensor
        Array<double> point_locs(2);  //nodal coordinate
        double theta_max;             //maximum shear direction wrt x coord

        //  
        point_locs[0] = locs[0 + stz_iter * 2];
        point_locs[1] = locs[1 + stz_iter * 2];
        
        sigma_total->eval(point_value,point_locs); //interpolate

        double a = abs(point_value[3] - point_value[0]);
        double b = abs(point_value[1]);

        if ( a > 1e9 * b ){
            theta_max = pi_ / 4;
            std::cout<<"CONDITON 1"<<std::endl;
        }
        else if ( b > 1e9 * a ){
            theta_max = 0;
            std::cout<<"CONDITON 2"<<std::endl;
        }
        else if ( a == 0 && b == 0 ){
            theta_max = 0;
            std::cout<<"CONDITON 3"<<std::endl;
        }
        else{
            theta_max = 0.5 * atan2((point_value[3] - point_value[0]) , ( 2.0 * point_value[1] )); //* 180.0 / pi_; //in degree; //if point_value[1] = 0, gives nan, need to make sure it doesn't affect results //atan2 returns [-pi,pi]
        }
        
        s_vec[0 + stz_iter * 2] =      cos(theta_max);  s_vec[1 + stz_iter * 2] = sin(theta_max);
        n_vec[0 + stz_iter * 2] = -1 * sin(theta_max);  n_vec[1 + stz_iter * 2] = cos(theta_max);

    }

    /*Compute coordinates for newly placed stzs once one existing stz is activated 
    //Goal: as one initial step only consider maximum shear direction, four stzs are placed (2 along s vec, 2 along n vec)
    //The distance between new stzs center and the existing activated stz is 2 * radius
    //Input: coordinate array (locs), s,n vector, stz index, array for storing new coordinates 
    //Output: modified "new_placed_stz_coord" vector */
    void new_stz_coord(std::vector<double> locs, std::vector<double> s_vec, std::vector<double> n_vec, int stz_iter, std::vector<double>& new_placed_stz_coord){

        //get stz coord
        auto stz_coordx = locs[0 + stz_iter * 2];
        auto stz_coordy = locs[1 + stz_iter * 2];

        //get corresponding s,n vector components
        auto stz_svec_1 = s_vec[0 + stz_iter * 2];
        auto stz_svec_2 = s_vec[1 + stz_iter * 2];

        auto stz_nvec_1 = n_vec[0 + stz_iter * 2];
        auto stz_nvec_2 = n_vec[1 + stz_iter * 2];

        //store new potential stz locations (there should be an overlapping check later)
        auto ptr1_x = stz_coordx + 2 * radius * stz_svec_1 + 2 * distol * stz_svec_1;  
        auto ptr1_y = stz_coordy + 2 * radius * stz_svec_2 + 2 * distol * stz_svec_2;
        
        auto ptr2_x = stz_coordx - 2 * radius * stz_svec_1 - 2 * distol * stz_svec_1;   
        auto ptr2_y = stz_coordy - 2 * radius * stz_svec_2 - 2 * distol * stz_svec_2;
        
        auto ptr3_x = stz_coordx + 2 * radius * stz_nvec_1 + 2 * distol * stz_nvec_1;  
        auto ptr3_y = stz_coordy + 2 * radius * stz_nvec_2 + 2 * distol * stz_nvec_2;
        
        auto ptr4_x = stz_coordx - 2 * radius * stz_nvec_1 - 2 * distol * stz_nvec_1;  
        auto ptr4_y = stz_coordy - 2 * radius * stz_nvec_2 - 2 * distol * stz_nvec_2;

        //Add exclusion zone free of STZ centers at the boundary
        if ( (ptr1_x >= 2 * radius) && (ptr1_y >= 2 * radius) && ((L - ptr1_x) >= 2 * radius) && ((h - ptr1_y) >= 2 * radius) ){
            new_placed_stz_coord.push_back(ptr1_x); new_placed_stz_coord.push_back(ptr1_y);
        }
        if ( (ptr2_x >= 2 * radius) && (ptr2_y >= 2 * radius) && ((L - ptr2_x) >= 2 * radius) && ((h - ptr2_y) >= 2 * radius) ){
            new_placed_stz_coord.push_back(ptr2_x); new_placed_stz_coord.push_back(ptr2_y);
        }
        if ( (ptr3_x >= 2 * radius) && (ptr3_y >= 2 * radius) && ((L - ptr3_x) >= 2 * radius) && ((h - ptr3_y) >= 2 * radius) ){
            new_placed_stz_coord.push_back(ptr3_x); new_placed_stz_coord.push_back(ptr3_y);
        }
        if ( (ptr4_x >= 2 * radius) && (ptr4_y >= 2 * radius) && ((L - ptr4_x) >= 2 * radius) && ((h - ptr4_y) >= 2 * radius) ){
            new_placed_stz_coord.push_back(ptr4_x); new_placed_stz_coord.push_back(ptr4_y);
        }

    }

    /*Adding new stzs and update size of arrays
    //Goal: As a initial step, only check the distance between two stzs: (x0 - x1) ^ 2 + (y0 - y1) ^ 2 ? (2a) ^ 2
    //Input: global array contains all potential new stz coordinate, locs of current existing stzs, size of stzs 
    //Output: new stzs will be added to "locs", the size "N0" will also be modified */
    void update_new_stz(std::vector<double> new_placed_stz_coord_global, std::vector<double>& locs){
        
        for (int i = 0; i < new_placed_stz_coord_global.size() / 2; i++ ){
            
            double new_stz_x = new_placed_stz_coord_global[0 + i * 2];
            double new_stz_y = new_placed_stz_coord_global[1 + i * 2];

            compare_stz(new_stz_x,new_stz_y,locs); //locs may be changed after this function //contains new stzs

        }
        /*
        //initialize arrays
        double stz_x, stz_y, stz_iter_x, stz_iter_y, dist;
        bool flag = true;
        std::vector<int> marker;
        std::vector<double> npscg_rm; //intermediate vector for storing new stz coordinates

        //check within new_placed_stz_coord_global
        for (int i = 0; i < new_placed_stz_coord_global.size() / 2; i++){
            stz_x = new_placed_stz_coord_global[0 + i * 2];
            stz_y = new_placed_stz_coord_global[1 + i * 2];
            for (int j = 0; j < new_placed_stz_coord_global.size() / 2; j++){ 
                flag = true;
                if ( ( i != j ) && ( j > i ) ){ //exclude self and do not repeat computing
                    stz_iter_x = new_placed_stz_coord_global[0 + j * 2];
                    stz_iter_y = new_placed_stz_coord_global[1 + j * 2];

                    dist_two_node(stz_x,stz_y,stz_iter_x,stz_iter_y,dist,flag);

                    if ( flag == false ){
                        marker.push_back(j);
                    } 

                }      
            }
        }

        std::sort( marker.begin(), marker.end() ); //sort
        marker.erase( unique( marker.begin(), marker.end() ), marker.end() ); //remove duplicates

        for (int k = 0; k < new_placed_stz_coord_global.size() / 2; k++){

            if (std::find(marker.begin(), marker.end(), k) != marker.end()){ 
                continue;
            }
            else{ //if marker does not contains k
                npscg_rm.push_back(new_placed_stz_coord_global[0 + k * 2]);
                npscg_rm.push_back(new_placed_stz_coord_global[1 + k * 2]);
            }

        }

        //Initialze arrays
        double npscg_x, npscg_y, locs_iter_x, locs_iter_y;
        marker.clear(); //reusue marker vector

        //check within existing stzs
        for (int l = 0; l < npscg_rm.size() / 2; l++){ //loop over intermediate new nodes
            
            npscg_x = npscg_rm[0 + l * 2];
            npscg_y = npscg_rm[1 + l * 2];

            flag = true; //if there is one existing stz less than 2a, the new stz is dropped
            
            for (int m = 0; m < locs.size() / 2; m++){ //loop over existing nodes

                locs_iter_x = locs[0 + m * 2];
                locs_iter_y = locs[1 + m * 2];

                dist_two_node(npscg_x,npscg_y,locs_iter_x,locs_iter_y,dist,flag);

            }

            if ( flag == false ){ //node overlaps
                //std::cout<<"overlap"<<l<<std::endl;
                marker.push_back(l);
            }

        }
        
        //finalize new stzs added in locs
        for (int n = 0; n < npscg_rm.size() / 2; n++){
            
            if (std::find(marker.begin(), marker.end(), n) != marker.end()){ //if marker does not contain n, add it to locs
                continue;
            }
            else{
                //std::cout<<"n:"<<n<<std::endl;
                locs.push_back(npscg_rm[0 + n * 2]);
                locs.push_back(npscg_rm[1 + n * 2]);
            }
        
        }

        //update N0: current number of stzs
        //N0 = locs.size() / 2;
        */

    }

    /*Helper function: compare within existing stzs */
    void compare_stz(double new_stz_x, double new_stz_y, std::vector<double>& locs){
        
        bool flag = true;
        double dist;

        for (int j = 0; j < locs.size() / 2; j++){

            double locs_stz_x = locs[0 + j * 2];
            double locs_stz_y = locs[1 + j * 2];

            dist_two_node(new_stz_x,new_stz_y,locs_stz_x,locs_stz_y,dist,flag);

        }

        if (flag == true)
        { 
            
            locs.push_back(new_stz_x);
            locs.push_back(new_stz_y);
        
        }

    }

    /*Helper function: compute distance between two nodes*/
    void dist_two_node(double node0_x, double node0_y, double node1_x, double node1_y, double& dist, bool& flag){
        
        dist = sqrt( pow( ( node0_x - node1_x ) , 2 ) + pow( ( node0_y - node1_y ) , 2 ) ); 

        if ( dist < 2 * (radius) ){ flag = false; } //if distance is less than 2 * radius, return false
    
    }

    /*Compute eigenstrain based on expression */
    double expression_eigenstrain(double t){
        return gamma_threshold_per_iter * ( 1 - std::exp( -t / B_ ) );
    }
};
int main(){
    
    //-------------------Read Mesh--------------------//
    //Define two corner points
    Point a0(x0, y0);
	Point a1(x1, y1);

    //Mesh FEM Simulation
    //###############TRIA3###############//
    auto mesh = std::make_shared<Mesh>(MPI_COMM_WORLD);
    auto mesh_box = std::make_shared<RectangleMesh>(a0,a1,nx,ny); //TRIA3 Structed Mesh
	mesh = mesh_box;
    //auto coord = mesh->coordinates();
    //###############QUAD4###############//
    //Note: mesh is a function, not a pointer
    //auto mesh = RectangleMesh::create(MPI_COMM_WORLD,{a0, a1}, {nx, ny}, CellType::Type::quadrilateral);

    //For Visualization
    //###############TRIA3###############//
    auto mesh_v = std::make_shared<Mesh>(MPI_COMM_WORLD);
    auto mesh_box_v = std::make_shared<RectangleMesh>(a0,a1,(10*nx),(10*ny)); //TRIA3 Structed Mesh
	mesh_v = mesh_box_v;
    //###############QUAD4###############//
    //Note: mesh is a function, not a pointer
    //auto mesh = RectangleMesh::create(MPI_COMM_WORLD,{a0, a1}, {20*nx, 20*ny}, CellType::Type::quadrilateral);

    //Get mesh coordinate
    //auto coord = mesh->coordinates();

    //Allocate Periodic Boundary
    auto periodic_boundary = std::make_shared<PeriodBC>(); //no periodic in this problem (JMPS) 

    //Define FunctionSpace
    //For FEM Simulation
    auto T2 = std::make_shared<tfunc2by2::FunctionSpace>(mesh);               //TensorFunctionSpace (2by2)
    auto W  = std::make_shared<vf::FunctionSpace>(mesh);                      //VectorFunctionSpace (2by1)

    //For analytical solution, no periodic condition
    auto T1 = std::make_shared<tfunc3by3::FunctionSpace>(mesh);                                 //TensorFunctionSpace (3by3)
    auto T2np = std::make_shared<tfunc2by2::FunctionSpace>(mesh);  
    auto Wnp = std::make_shared<vf::FunctionSpace>(mesh);                                       //VectorFunctionSpace (2by1) 

    auto X  = std::make_shared<scalarfunc::FunctionSpace>(mesh);              //FunctionSpace       (1by1)

    //For Visualization
    auto T1v = std::make_shared<tfunc3by3::FunctionSpace>(mesh_v);            //TensorFunctionSpace (3by3)
    auto T2v = std::make_shared<visual_eps::FunctionSpace>(mesh_v);           //TensorFunctionSpace (2by2) 

    //Check
    /*
    //File mfile("res/mesh.pvd");
    //mfile << *mesh_quad;
    */
    //------------------------------------------------//

    //----------Initialize Problem Boundaries----------//
    auto boundary_left   = std::make_shared<BC_left>();
    auto boundary_right  = std::make_shared<BC_right>();
    auto boundary_top    = std::make_shared<BC_top>();
    auto boundary_bottom = std::make_shared<BC_bottom>();
    auto boundary_point  = std::make_shared<BC_point>();

    //Define Meshfunction for Boundaries
    auto boundaries = std::make_shared<MeshFunction<std::size_t>>(mesh, mesh->topology().dim()-1);     //meshfunction for facets
    //auto boundaries_ptr = std::make_shared<MeshFunction<std::size_t>>(mesh, mesh->topology().dim()-2); //meshfunction for points
    
    *boundaries = 0;
    boundary_top->mark(*boundaries, 1);      // 1
    boundary_bottom->mark(*boundaries, 2);   // 2
    boundary_left->mark(*boundaries, 3);     // 3
    boundary_right->mark(*boundaries, 4);    // 4

    //*boundaries_ptr = 0;
    //boundary_point->mark(*boundaries_ptr, 5);    // 5


    /*Check boundaries*/
    //File bcfile ("test/bc_marker.pvd");
    //bcfile << *boundaries_ptr;

    ///*Check periodic bc using master-slaves function (pass)
    //auto pb_comp_ptr = std::make_shared<PeriodicBoundaryComputation>();
    //auto meshfunc = pb_comp_ptr->masters_slaves(mesh, *periodic_boundary, 1); //1: edge 0: vertex 2:facet
    //Check
    //File bfile("test/meshfunc_marker.pvd");
	//bfile << meshfunc;
    //*/
    
    //------------------------------------------------//

    //----------Define Function & Assign BC-----------//
    //*DISPLACEMENT*//
    //w periodic
    auto u_image = std::make_shared<Function>(W);          //image displacement vector (global)
    auto u_image_step = std::make_shared<Function>(W);     //image displacement step vector (global)
    
    //w/o periodic
    auto u_total = std::make_shared<Function>(W);          //total displacement vector (global)
    auto u_tilde = std::make_shared<Function>(W);          //tilde displacement vector (global)
    auto u_tilde_step = std::make_shared<Function>(W);     //tilde displacement vector step (global)

    auto u_tilde_sum = std::make_shared<Function>(W);      //tilde displacement rate vector (global)
    auto u_tilde_stz = std::make_shared<Function>(W);      //tilde displacement rate vector (stz)
    //auto u_tilde_sum_id3 = std::make_shared<Function>(Wnp);

    auto u_image_periodic_to_full = std::make_shared<Function>(W); //transform periodic image field to total field

    /*STRAIN*/
    //w periodic
    auto eps_image = std::make_shared<Function>(T2);         //image strain tensor (global)
    auto eps_image_step = std::make_shared<Function>(T2);    //image strain step tensore (global)

    //w/o periodic
    auto eps_total = std::make_shared<Function>(T2);       //total strain tensor (global)
    auto eps_tilde = std::make_shared<Function>(T2);       //tilde strain tensor (global)
    auto eps_tilde_step = std::make_shared<Function>(T2);  //tilde strain step tensor (global)

    auto eps_tilde_sum = std::make_shared<Function>(T2);   //tilde strain rate tensor (global) 
    auto eps_tilde_stz = std::make_shared<Function>(T2);   //tilde strain rate tensor (stz)
    //auto eps_tilde_sum_id3 = std::make_shared<Function>(T2np); 

    /*STRESS*/
    //w periodic
    auto sts_image = std::make_shared<Function>(T2);    //image stress tensor (global)

    //w/o periodic
    auto sts_total = std::make_shared<Function>(T2);       //total stress tensor (global)
    auto sts_tilde = std::make_shared<Function>(T2);       //tilde stress tensor (global) 
    auto sts_tilde_step = std::make_shared<Function>(T2);  //tilde stress step tensor (global)

    auto sts_tilde_sum = std::make_shared<Function>(T2);   //tilde stress rate tensor (global)
    auto sts_tilde_stz = std::make_shared<Function>(T2);   //tilde stress rate tensor (stz)
    //auto sts_tilde_sum_id3 = std::make_shared<Function>(T2np);

    /*ENERGY*/
    auto phi_engs  = std::make_shared<Function>(W);     //shear strain energy scalar (global)

    /*DMAT,HMAT*/
    auto Dmat = std::make_shared<Function>(T1);         //Dmat (stz)
    auto Hmat = std::make_shared<Function>(T1); 
    
    /*VISUALIZATION*/
    /*TILDE FIELD*/
    auto eps_tilde_v = std::make_shared<Function>(T2v); //tilde strain tensor (global)

    /*TOTAL FIELD*/
    auto eps_total_v = std::make_shared<Function>(T2v); //total strain tensor (global)
    auto sts_total_v = std::make_shared<Function>(T2v); //total stress tensor (global)

    //Define Boundary Conditions
    auto axial_tension = std::make_shared<axialtension>(u_tilde_step);
    auto zero_scalar_x = std::make_shared<zero_scalarx>(u_tilde_step);
    auto zero_scalar_y = std::make_shared<zero_scalary>(u_tilde_step);

    //auto zero_scalar = std::make_shared<Constant>(0.0);
    //auto zero_vector = std::make_shared<Constant>(0.0,0.0);
    
    //Declare Dirichlet Boundary Condtion
    DirichletBC u_tension (W->sub(1),         axial_tension,    boundary_top);              //axial tension in y dir
    //DirichletBC u_top     (W->sub(1),         zero_scalar_y,    boundary_top); 
    DirichletBC u_bot_x   (W->sub(0),         zero_scalar_x, boundary_point, "pointwise");  //zero displacement in x dir at one bottom point (enforce no rigid body motion)
    DirichletBC u_bot_y   (W->sub(1),         zero_scalar_y, boundary_bottom);              //zero displacement in y dir at the bottom
    std::vector<DirichletBC*> bc_u = {{&u_tension,&u_bot_x,&u_bot_y}};
    //std::vector<DirichletBC*> bc_u = {{&u_tension,&u_bot_y}};
    //------------------------------------------------//

    //-----Define variational form & ufl mapping------//
    
    //For solving displs image field (see vf.ufl)
    auto E_ufl  = std::make_shared<Constant>(E);
    auto nu_ufl = std::make_shared<Constant>(nu);

    auto F_u = std::make_shared<vf::LinearForm>(W);
    auto J_u = std::make_shared<vf::JacobianForm>(W,W);

    std::map<std::string, std::shared_ptr<const GenericFunction>> coefficients_F_u = \
    {{"u1",u_image_step},{"E",E_ufl},{"nu",nu_ufl},{"sigma_tilde",sts_tilde_step}};
    std::map<std::string, std::shared_ptr<const GenericFunction>> coefficients_J_u = \
    {{"E",E_ufl},{"nu",nu_ufl}};

    F_u->set_coefficients(coefficients_F_u);
    J_u->set_coefficients(coefficients_J_u);

    F_u->ds = boundaries;
    J_u->ds = boundaries;

    //For post-processing compute strain field (see project.ufl)
    auto F_post_eps = std::make_shared<project::LinearForm>(T2);
    auto J_post_eps = std::make_shared<project::JacobianForm>(T2,T2);

    std::map<std::string, std::shared_ptr<const GenericFunction>> coefficients_F_post_eps = \
    {{"eps",eps_image_step},{"u",u_image_step}};
    std::map<std::string, std::shared_ptr<const GenericFunction>> coefficients_J_post_eps = \
    {};

    F_post_eps->set_coefficients(coefficients_F_post_eps);
    J_post_eps->set_coefficients(coefficients_J_post_eps);

    //For visualization (see visual_eps.ufl)
    //tip: to check for coefficients, search "coefficient_number" in header file
    auto F_eps_tilde = std::make_shared<visual_eps::LinearForm>(T2v);
    auto J_eps_tilde = std::make_shared<visual_eps::JacobianForm>(T2v,T2v);
    
    std::map<std::string, std::shared_ptr<const GenericFunction>> coefficients_Fepsv = \
    {{"eps_coarse",eps_total},{"eps_fine",eps_total_v}};
    std::map<std::string, std::shared_ptr<const GenericFunction>> coefficients_Jepsv = \
    {};
    
    F_eps_tilde->set_coefficients(coefficients_Fepsv);
    J_eps_tilde->set_coefficients(coefficients_Jepsv);

    auto F_sts_cf = std::make_shared<visual_sts::LinearForm>(T2v);
    auto J_sts_cf = std::make_shared<visual_sts::JacobianForm>(T2v,T2v);

    std::map<std::string, std::shared_ptr<const GenericFunction>> coefficients_Fstsv = \
    {{"sts_coarse",sts_total},{"sts_fine",sts_total_v}};
    std::map<std::string, std::shared_ptr<const GenericFunction>> coefficients_Jstsv = \
    {};

    F_sts_cf->set_coefficients(coefficients_Fstsv);
    J_sts_cf->set_coefficients(coefficients_Jstsv);

    //-------------User Defined Solver----------------//
    /*SOLVE DISPLS IMAGE FIELD*/
    auto solver_vf = std::make_shared<PETScKrylovSolver>("gmres");
    KSP ksp_vf = solver_vf->ksp();
    PC pc_vf;
    NSequation nseq_eps_vf(F_u, J_u, bc_u);
    KSPGetPC(ksp_vf, &pc_vf);
    dolfin::PETScOptions::set("pc_type", "hypre");
    KSPSetFromOptions(ksp_vf);
    NewtonSolver ns_vf(MPI_COMM_WORLD, solver_vf, PETScFactory::instance());
    ns_vf.parameters["relative_tolerance"] = 1.0e-8;  //1e-8
	ns_vf.parameters["absolute_tolerance"] = 1.0e-10; //1e-10
	ns_vf.parameters["maximum_iterations"] = 30;
    ns_vf.parameters["error_on_nonconvergence"] = false;

    /*PROJECTION*/
    auto solver_project = std::make_shared<PETScKrylovSolver>("gmres");
    KSP ksp_project = solver_project->ksp();
    PC pc_project;
    NSequation nseq_eps_project(F_post_eps, J_post_eps, {});
    KSPGetPC(ksp_project, &pc_project);
    dolfin::PETScOptions::set("pc_type", "hypre");
    KSPSetFromOptions(ksp_project);
    NewtonSolver ns_project(MPI_COMM_WORLD, solver_project, PETScFactory::instance());
	ns_project.parameters["relative_tolerance"] = 1.0e-10;
	ns_project.parameters["absolute_tolerance"] = 1.0e-15;
	ns_project.parameters["maximum_iterations"] = 10;

    /*VISUALIZATION*/
    auto solver_v = std::make_shared<PETScKrylovSolver>("gmres");
    KSP ksp_v = solver_v->ksp();
    PC pc_v;
    NSequation nseq_eps_v(F_eps_tilde, J_eps_tilde, {});
    KSPGetPC(ksp_v, &pc_v);
    dolfin::PETScOptions::set("pc_type", "hypre");
    KSPSetFromOptions(ksp_v);
    NewtonSolver ns_v(MPI_COMM_WORLD, solver_v, PETScFactory::instance());
	ns_v.parameters["relative_tolerance"] = 1.0e-30;
	ns_v.parameters["absolute_tolerance"] = 1.0e-30;
	ns_v.parameters["maximum_iterations"] = 10;

    auto solver_v_sts = std::make_shared<PETScKrylovSolver>("gmres");
    KSP ksp_v_sts = solver_v_sts->ksp();
    PC pc_v_sts;
    NSequation nseq_eps_v_sts(F_sts_cf, J_sts_cf, {});
    KSPGetPC(ksp_v_sts, &pc_v_sts);
    dolfin::PETScOptions::set("pc_type", "hypre");
    KSPSetFromOptions(ksp_v_sts);
    NewtonSolver ns_v_sts(MPI_COMM_WORLD, solver_v_sts, PETScFactory::instance());
	ns_v_sts.parameters["relative_tolerance"] = 1.0e-25;
	ns_v_sts.parameters["absolute_tolerance"] = 1.0e-25;
	ns_v_sts.parameters["maximum_iterations"] = 10;
    

    //------------------Initialization----------------//
    
    //stz_helper class
    auto stz_class = std::make_shared<stz_helper>(); 
    
    //Define flat 2D array like vector objects for storing items 
    //array size has to be known at runtime 
    std::vector<double> locs            (N0*2, -__DBL_MAX__);    //array for storing locs : (x,y) coordinate
    std::vector<double> engs_c          (N0*1, -__DBL_MAX__);    //array for storing engs : !critical! shear strain energy
    std::vector<double> engs            (N0*1, -__DBL_MAX__);    //array for storing engs : !current!  shear strain energy
    std::vector<double> svec            (N0*2, -__DBL_MAX__);    //array for storing svec : parallel to maximum shear direction
    std::vector<double> nvec            (N0*2, -__DBL_MAX__);    //array for storing nvec : perpendicular to maximum shear direction
    std::vector<double> configf         (N0*1, -__DBL_MAX__);    //array for storing configf : configurational force 
    std::vector<double> gammadot        (N0*1, -__DBL_MAX__);    //array for storing gammadot : rate of shear strain
    std::vector<double> eigen_eps       (N0*4, -__DBL_MAX__);    //array for storing eigenstrain rate (specified at each stz location)
    std::vector<double> gamma_vec       (N0*1,          0.0);    //array for storing gamma value: shear strain in local coordinate
    std::vector<int>    active_status   (N0*1,            0);    //array for storing current activation status: 0:not exceed threshold 1:activation cont 2:activation close; ., use MPI_IN_PLACE and MPI_MAX
    std::vector<double> gamma_step_vec  (N0*1,          0.0);    //array for storing step gamma value (incremental analysis)
    std::vector<double> eigen_eps_mpi   (N0*4, -__DBL_MAX__);    //array for storing eigenstrain rate after allreduce call

    std::vector<int>    active_status_mpi (N0*1,          0);    //array for storing global active status
    std::vector<double> gamma_vec_mpi     (N0*1,          0);    

    std::vector<int> activated_stz_vec;                        //array for storing activated stzs index
    int activated_stz_local_count = 0;                         //number of activated stzs belongs to local processor
    std::vector<int> activated_stz_global_index;               //index of activated stzs (here for init purpose only)
    std::vector<int> activated_stz_global_count(rank_num, 0);  //receive buffer for number of activated stzs
    std::vector<int> activated_stz_displs(rank_num, 0);        //displs buffer for sending activated stzs index

    std::vector<int> reactivate_indicator(N0*1,  -INT_MAX);    //array for record permission for reactivation

    std::vector<int> reactivate_indicator_mpi(N0*1,  -INT_MAX);    //array for record permission for reactivation

    std::vector<double> new_placed_stz_coord;                  //array for storing newly placed stzs (for local processor)
    int new_stz_count_local;                                   //number of new stzs for each local processor
    bool new_stz_flag = false;                                 //flag indicates whether there are new stzs
    std::vector<int> new_stz_count_global(rank_num, 0);        //vector contains number of new stzs from each processor
    std::vector<int> new_stz_displs(rank_num, 0);              //displs buffer for contains coordinate of new stzs

    //Add vectors for volumetric Eshley part
    std::vector<double> epsilondot       (N0*1, -__DBL_MAX__);    //array for storing gammadot : rate of tensile strain
    std::vector<double> epsilon_vec      (N0*1,          0.0);    //array for storing gamma value: tensile strain in local coordinate
    std::vector<double> epsilon_vec_mpi  (N0*1,            0);
    std::vector<double> epsilon_step_vec (N0*1,          0.0);    //array for storing step gamma value (incremental analysis)
    std::vector<double> configf_v        (N0*1, -__DBL_MAX__);    //array for storing configf : configurational force for volumetric strain
    std::vector<double> eigen_eps_v      (N0*4, -__DBL_MAX__);    //array for storing eigenstrain rate (specified at each stz location)
    std::vector<double> eigen_eps_mpiv   (N0*4, -__DBL_MAX__);    //array for storing eigenstrain rate after allreduce call

    //Initial STZ locs & shear strain energy
    stz_class->stz_init(ij, kl, N0, locs, engs_c, 0, N_new);

    //Use c++ built-in
    //stz_class->random_gaussian_distribution(engs_c,0,N0,N_new);

    //Use c++ built-in
    //stz_class->random_uniform_distribution(locs, 0, N0, N_new);

    /***************************************/
    //Change location to be center of domain
    //locs[0] = L / 2.0; locs[1] = h / 2.0;
    /***************************************/

    //std::cout<<"coord:"<<locs[0]<<" "<<locs[1]<<std::endl;

    //Open txt file
    std::ofstream myfile;  //txt stores stress strain curve
    myfile.open ("stress_strain.txt");

    std::ofstream stzlocs; //txt stores stz locations
    stzlocs.open ("stzlocs.txt");

    std::ofstream actindx; //txt stores activated index
    actindx.open ("actindex.txt");

    std::ofstream engcsave;
    engcsave.open ("engcsave.txt");

    std::ofstream mid_data; //txt store partial stz locs & activate status
    std::ofstream mid_curve;
    std::ofstream mid_reactive;

    myfile << std::to_string(0.0) + " " + std::to_string(0.0) + "\n";

    mid_curve.open ("curve/mid_curve.txt",std::ios::app);
    mid_curve << std::to_string(0.0) + " " + std::to_string(0.0) + "\n";
    mid_curve.close();
    
    std::cout<<"SIZE of engs_c: "<<engs_c.size()<<std::endl;
    std::cout<<"MIN eng_c:"<<*std::min_element(engs_c.begin(),engs_c.end())<<std::endl;
    std::cout<<"MAX eng_c:"<<*std::max_element(engs_c.begin(),engs_c.end())<<std::endl;
    for (int i = 0; i < engs_c.size(); i++){
        //std::cout<<engs_c[i]<<std::endl;
        engcsave<<engs_c[i]<<std::endl;
    }

    if (MPI::rank(MPI_COMM_WORLD) == 0){
        for (int i = 0; i < locs.size() / 2; i++){
            stzlocs << std::to_string(locs[0 + i * 2])<<" "<<std::to_string(locs[1 + i * 2]) + "\n";     
        }
    }
    
    //-------------------Main Loop--------------------//
    double t_cur = 0.0; //initialize current time
    int step_num = 0;   //initialize step count

    //-------------Identity (locs,processor) relation-----------//
    //Pre-loop stz locs
    Array<double> point_locs_preloop(2); 
    //Get bounding tree
    auto bbt = mesh->bounding_box_tree();
    //Initialize marker for storing (locs,processor) relation
    std::vector<int> locs_ownership_processor(1e5, -1);
    std::vector<int> locs_ownership;
    
    for (int i = 0; i < N0; i++){
        point_locs_preloop[0] = locs[0 + i * 2];
        point_locs_preloop[1] = locs[1 + i * 2];
        if (bbt->collides_entity(point_locs_preloop)){ //this returns a boolean type: true if point is within the local mesh
            //std::cout<<MPI::rank(MPI_COMM_WORLD)<<std::endl;
            locs_ownership_processor[i] = MPI::rank(MPI_COMM_WORLD); //store the processor rank
            //locs_ownership.push_back(i);
        }
    }

    
    MPI_Allreduce(MPI_IN_PLACE, locs_ownership_processor.data(), locs_ownership_processor.size(), MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    
    //if (MPI::rank(MPI_COMM_WORLD) == 0){
    //    for (int i = 0; i < 30; i++){
    //        std::cout<<locs_ownership_processor[i]<<std::endl;
    //    }
    //}

    for (int i = 0; i < N0; i++){
        if ( locs_ownership_processor[i] == MPI::rank(MPI_COMM_WORLD) ){
            locs_ownership.push_back(i);
        }
    }

    std::cout<<"INIT LOCS_OWNERSHIP DONE !"<<std::endl;

    std::cout<<locs[-2]<<" "<<locs[-1]<<std::endl;

    /*Check locs_ownership
    if (MPI::rank(MPI_COMM_WORLD) == 0){
        for (int j = 0; j < locs_ownership.size(); j++){
            std::cout<<locs_ownership[j]<<std::endl;
        }
    }
    */
    
    int dof_loc;
    std::vector<double>  Dmat1111_stzs_vec, Dmat1122_stzs_vec, Dmat1112_stzs_vec, Dmat2211_stzs_vec, Dmat2222_stzs_vec, Dmat2212_stzs_vec, Dmat1211_stzs_vec, Dmat1222_stzs_vec, Dmat1212_stzs_vec;
    std::vector<double>  Hmat111_stzs_vec, Hmat122_stzs_vec, Hmat112_stzs_vec, Hmat211_stzs_vec, Hmat222_stzs_vec, Hmat212_stzs_vec;

    //Compute Dmat/Hmat for initial stzs
    for (int stz_init = 0; stz_init < N_new; stz_init++){ //loop over stzs

        if (MPI::rank(MPI_COMM_WORLD) == 0){
            std::cout<<"INIT DMAT/HMAT VEC PROGRESS:"<<" "<<(1.0*stz_init)/(1.0*N_new) * 100<<"%"<<std::endl;
        }

        double stz_init_coords[2]; //get coord
        std::vector<double> D1111_vec, D1122_vec, D1112_vec, D2211_vec, D2222_vec, D2212_vec, D1211_vec, D1222_vec, D1212_vec; //for storing Dmat tensor components, different from local processors

        stz_init_coords[0] = locs[0 + stz_init * 2];
        stz_init_coords[1] = locs[1 + stz_init * 2];

        auto Dclass = std::make_shared<tensor_D>(stz_init_coords);
        *Dmat = *Dclass;

        auto D1111_ptr = std::make_shared<Function>((*Dmat)[0]);
        auto D1122_ptr = std::make_shared<Function>((*Dmat)[1]);
        auto D1112_ptr = std::make_shared<Function>((*Dmat)[2]);
        auto D2211_ptr = std::make_shared<Function>((*Dmat)[3]);
        auto D2222_ptr = std::make_shared<Function>((*Dmat)[4]);
        auto D2212_ptr = std::make_shared<Function>((*Dmat)[5]);
        auto D1211_ptr = std::make_shared<Function>((*Dmat)[6]);
        auto D1222_ptr = std::make_shared<Function>((*Dmat)[7]);
        auto D1212_ptr = std::make_shared<Function>((*Dmat)[8]);

        D1111_ptr->vector()->get_local(D1111_vec);
        D1122_ptr->vector()->get_local(D1122_vec);
        D1112_ptr->vector()->get_local(D1112_vec);
        D2211_ptr->vector()->get_local(D2211_vec);
        D2222_ptr->vector()->get_local(D2222_vec);
        D2212_ptr->vector()->get_local(D2212_vec);
        D1211_ptr->vector()->get_local(D1211_vec);
        D1222_ptr->vector()->get_local(D1222_vec);
        D1212_ptr->vector()->get_local(D1212_vec);

        auto ndof = D1111_vec.size(); dof_loc = ndof;

        if ( stz_init == 0 ){
            Dmat1111_stzs_vec.resize(ndof * N_new, 0.0);  //arrays for storing Dmat local vector for each stzs;
            Dmat1122_stzs_vec.resize(ndof * N_new, 0.0); 
            Dmat1112_stzs_vec.resize(ndof * N_new, 0.0); 
            Dmat2211_stzs_vec.resize(ndof * N_new, 0.0); 
            Dmat2222_stzs_vec.resize(ndof * N_new, 0.0); 
            Dmat2212_stzs_vec.resize(ndof * N_new, 0.0); 
            Dmat1211_stzs_vec.resize(ndof * N_new, 0.0); 
            Dmat1222_stzs_vec.resize(ndof * N_new, 0.0); 
            Dmat1212_stzs_vec.resize(ndof * N_new, 0.0);
        }

        for (int j = 0; j < ndof; j++){ //store values to vector 
            Dmat1111_stzs_vec[ j + stz_init * ndof ] = D1111_vec[j];
            Dmat1122_stzs_vec[ j + stz_init * ndof ] = D1122_vec[j];
            Dmat1112_stzs_vec[ j + stz_init * ndof ] = D1112_vec[j];
            Dmat2211_stzs_vec[ j + stz_init * ndof ] = D2211_vec[j];
            Dmat2222_stzs_vec[ j + stz_init * ndof ] = D2222_vec[j];
            Dmat2212_stzs_vec[ j + stz_init * ndof ] = D2212_vec[j];
            Dmat1211_stzs_vec[ j + stz_init * ndof ] = D1211_vec[j];
            Dmat1222_stzs_vec[ j + stz_init * ndof ] = D1222_vec[j];
            Dmat1212_stzs_vec[ j + stz_init * ndof ] = D1212_vec[j];
        }

        std::vector<double> H111_vec, H122_vec, H112_vec, H211_vec, H222_vec, H212_vec;

        auto Hclass = std::make_shared<tensor_H>(stz_init_coords);
        *Hmat = *Hclass;

        auto H111_ptr = std::make_shared<Function>((*Hmat)[0]);
        auto H122_ptr = std::make_shared<Function>((*Hmat)[1]);
        auto H112_ptr = std::make_shared<Function>((*Hmat)[2]);
        auto H211_ptr = std::make_shared<Function>((*Hmat)[3]);
        auto H222_ptr = std::make_shared<Function>((*Hmat)[4]);
        auto H212_ptr = std::make_shared<Function>((*Hmat)[5]);

        H111_ptr->vector()->get_local(H111_vec);
        H122_ptr->vector()->get_local(H122_vec);
        H112_ptr->vector()->get_local(H112_vec);
        H211_ptr->vector()->get_local(H211_vec);
        H222_ptr->vector()->get_local(H222_vec);
        H212_ptr->vector()->get_local(H212_vec);

        if ( stz_init == 0 ){
            Hmat111_stzs_vec.resize(ndof * N_new, 0.0);  //arrays for storing Dmat local vector for each stzs;
            Hmat122_stzs_vec.resize(ndof * N_new, 0.0); 
            Hmat112_stzs_vec.resize(ndof * N_new, 0.0); 
            Hmat211_stzs_vec.resize(ndof * N_new, 0.0); 
            Hmat222_stzs_vec.resize(ndof * N_new, 0.0); 
            Hmat212_stzs_vec.resize(ndof * N_new, 0.0); 
        }

        for (int j = 0; j < ndof; j++){ //store values to vector 
            Hmat111_stzs_vec[ j + stz_init * ndof ] = H111_vec[j];
            Hmat122_stzs_vec[ j + stz_init * ndof ] = H122_vec[j];
            Hmat112_stzs_vec[ j + stz_init * ndof ] = H112_vec[j];
            Hmat211_stzs_vec[ j + stz_init * ndof ] = H211_vec[j];
            Hmat222_stzs_vec[ j + stz_init * ndof ] = H222_vec[j];
            Hmat212_stzs_vec[ j + stz_init * ndof ] = H212_vec[j];
        }         

    }

    //----------------------Main Loop---------------------------//

    double dt_current;
    while ( t_cur < T ){
        
        if ( t_cur < 1.35e-1 ){
            dt_current = dt;
        }
        else{
            dt_current = dt_small;
            dt_flag = 2;
        }
        t_cur += dt_current;


        //Output Time
        if (MPI::rank(MPI_COMM_WORLD) == 0){
            std::cout << "###########################" << std::endl;
            std::cout << "Current Time (sec):"<< t_cur << std::endl;
            std::cout << "###########################" << std::endl;
        }

        //Update step
        step_num += 1;
        
        //Compute shear strain energy (global)
        //!May need to modify here to include volumertic change!
        stz_class->strain_energy(sts_total, phi_engs);

        //Reinitialize arrays
        new_placed_stz_coord.clear(); //new stz coords
        activated_stz_vec.clear();    //index for activated stz

        //############ Update new stzs ###################//
        if ( new_stz_flag == true ){
            
            stz_class->stz_init(ij, kl, N0, locs, engs_c, 1, N_new); //->engs_c

            //use c++ built in
            //stz_class->random_gaussian_distribution(engs_c,1,N0,N_new); //->engs_c
            
            if (N_new - N0 <= 0){ std::cout<<"WRONG"<<std::endl; }
            
            //define initialization vectors
            int Nreinit_scalar = (N_new-N0) * 1;    std::vector<double> reinit_scalar(Nreinit_scalar, -__DBL_MAX__);
            int Nreinit_vector = (N_new-N0) * 2;    std::vector<double> reinit_vector(Nreinit_vector, -__DBL_MAX__);
            int Nreinit_tensor = (N_new-N0) * 4;    std::vector<double> reinit_tensor(Nreinit_tensor, -__DBL_MAX__);
            
            int Nreinit_scalard0 = (N_new-N0) * 1;  std::vector<double> reinit_scalard0(Nreinit_scalard0, 0.0);
            int Nreinit_scalar0  = (N_new-N0) * 1;  std::vector<int>    reinit_scalar0(Nreinit_scalar0, 0);

            int Nreinit_scalari0 = (N_new-N0) * 1;  std::vector<int> reinit_scalari0(Nreinit_scalari0, -INT_MAX);
            //update each quantity vectors
            engs.insert(engs.end(),reinit_scalar.begin(),reinit_scalar.end()); //->engs (scalar)
            svec.insert(svec.end(),reinit_vector.begin(),reinit_vector.end()); //->svec (vector)
            nvec.insert(nvec.end(),reinit_vector.begin(),reinit_vector.end()); //->nvec (vector)
            configf.insert(configf.end(),reinit_scalar.begin(),reinit_scalar.end()); //->config (scalar)
            gammadot.insert(gammadot.end(),reinit_scalar.begin(),reinit_scalar.end()); //->gammadot (scalar)
            eigen_eps.insert(eigen_eps.end(),reinit_tensor.begin(),reinit_tensor.end()); //->eigen_eps_rate (tensor)
            eigen_eps_mpi.insert(eigen_eps_mpi.end(),reinit_tensor.begin(),reinit_tensor.end()); //->eigen_eps_rate_mpi (tensor)
            
            gamma_vec.insert(gamma_vec.end(),reinit_scalard0.begin(),reinit_scalard0.end());//->gamma_vec (scalar, double, 0.0)
            gamma_step_vec.insert(gamma_step_vec.end(),reinit_scalard0.begin(),reinit_scalard0.end());//->gamma_step_vec (scalar, double, 0.0)
            gamma_vec_mpi.insert(gamma_vec_mpi.end(),reinit_scalard0.begin(),reinit_scalard0.end());
            
            active_status.insert(active_status.end(),reinit_scalar0.begin(),reinit_scalar0.end());//->active_status (scalar, int, 0)
            active_status_mpi.insert(active_status_mpi.end(),reinit_scalar0.begin(),reinit_scalar0.end());
            
            reactivate_indicator.insert(reactivate_indicator.end(),reinit_scalari0.begin(),reinit_scalari0.end());
            reactivate_indicator_mpi.insert(reactivate_indicator_mpi.end(),reinit_scalari0.begin(),reinit_scalari0.end());

            //update local_ownership for local processor
            for (int i = N0; i < N_new; i++){
                point_locs_preloop[0] = locs[0 + i * 2];
                point_locs_preloop[1] = locs[1 + i * 2];
                if (bbt->collides_entity(point_locs_preloop)){ //this returns a boolean type: true if point is within the local mesh
                    //std::cout<<MPI::rank(MPI_COMM_WORLD)<<std::endl;
                    locs_ownership_processor[i] = MPI::rank(MPI_COMM_WORLD); //store the processor rank
                    //locs_ownership.push_back(i);
                }
            }

            //std::cout<<"Check active_status"<<std::endl;
            //for ( auto item : active_status){
            //    std::cout<<item<<std::endl;
            //}

            
            MPI_Allreduce(MPI_IN_PLACE, locs_ownership_processor.data(), locs_ownership_processor.size(), MPI_INT, MPI_MAX, MPI_COMM_WORLD);

            for (int i = N0; i < N_new; i++){
                if ( locs_ownership_processor[i] == MPI::rank(MPI_COMM_WORLD) ){
                    locs_ownership.push_back(i);
                }
            }

            //update Dmat, Hmat for new added stz
            int Nreinit_scalar_mat = (N_new-N0) * dof_loc;  std::vector<double> reinit_scalar_mat(Nreinit_scalar_mat, 0.0); //Each stz gives vec of size dof_loc

            Dmat1111_stzs_vec.insert(Dmat1111_stzs_vec.end(),reinit_scalar_mat.begin(),reinit_scalar_mat.end());  //arrays for storing Dmat local vector for each stzs;
            Dmat1122_stzs_vec.insert(Dmat1122_stzs_vec.end(),reinit_scalar_mat.begin(),reinit_scalar_mat.end()); 
            Dmat1112_stzs_vec.insert(Dmat1112_stzs_vec.end(),reinit_scalar_mat.begin(),reinit_scalar_mat.end()); 
            Dmat2211_stzs_vec.insert(Dmat2211_stzs_vec.end(),reinit_scalar_mat.begin(),reinit_scalar_mat.end()); 
            Dmat2222_stzs_vec.insert(Dmat2222_stzs_vec.end(),reinit_scalar_mat.begin(),reinit_scalar_mat.end()); 
            Dmat2212_stzs_vec.insert(Dmat2212_stzs_vec.end(),reinit_scalar_mat.begin(),reinit_scalar_mat.end()); 
            Dmat1211_stzs_vec.insert(Dmat1211_stzs_vec.end(),reinit_scalar_mat.begin(),reinit_scalar_mat.end()); 
            Dmat1222_stzs_vec.insert(Dmat1222_stzs_vec.end(),reinit_scalar_mat.begin(),reinit_scalar_mat.end()); 
            Dmat1212_stzs_vec.insert(Dmat1212_stzs_vec.end(),reinit_scalar_mat.begin(),reinit_scalar_mat.end());
            
            Hmat111_stzs_vec.insert(Hmat111_stzs_vec.end(),reinit_scalar_mat.begin(),reinit_scalar_mat.end());  //arrays for storing Dmat local vector for each stzs;
            Hmat122_stzs_vec.insert(Hmat122_stzs_vec.end(),reinit_scalar_mat.begin(),reinit_scalar_mat.end()); 
            Hmat112_stzs_vec.insert(Hmat112_stzs_vec.end(),reinit_scalar_mat.begin(),reinit_scalar_mat.end()); 
            Hmat211_stzs_vec.insert(Hmat211_stzs_vec.end(),reinit_scalar_mat.begin(),reinit_scalar_mat.end()); 
            Hmat222_stzs_vec.insert(Hmat222_stzs_vec.end(),reinit_scalar_mat.begin(),reinit_scalar_mat.end()); 
            Hmat212_stzs_vec.insert(Hmat212_stzs_vec.end(),reinit_scalar_mat.begin(),reinit_scalar_mat.end()); 

            for (int stz_addnew = N0; stz_addnew < N_new; stz_addnew++){ 

                if (MPI::rank(MPI_COMM_WORLD) == 0 && N_new > N0){
                    //std::cout<<"ADDED NEW STZ NUMBER: "<<" "<<N_new - N0<<std::endl;
                    std::cout<<"ADDED DMAT/HMAT VEC PROGRESS:"<<" "<<(1.0*(stz_addnew-N0))/(1.0*(N_new-N0)) * 100<<"%"<<std::endl;
                }

                double stz_add_coords[2]; //get coord
                std::vector<double> D1111_vec, D1122_vec, D1112_vec, D2211_vec, D2222_vec, D2212_vec, D1211_vec, D1222_vec, D1212_vec; //for storing Dmat tensor components, different from local processors

                stz_add_coords[0] = locs[0 + stz_addnew * 2];
                stz_add_coords[1] = locs[1 + stz_addnew * 2];

                auto Dclass = std::make_shared<tensor_D>(stz_add_coords);
                *Dmat = *Dclass;

                auto D1111_ptr = std::make_shared<Function>((*Dmat)[0]);
                auto D1122_ptr = std::make_shared<Function>((*Dmat)[1]);                
                auto D1112_ptr = std::make_shared<Function>((*Dmat)[2]);
                auto D2211_ptr = std::make_shared<Function>((*Dmat)[3]);
                auto D2222_ptr = std::make_shared<Function>((*Dmat)[4]);
                auto D2212_ptr = std::make_shared<Function>((*Dmat)[5]);
                auto D1211_ptr = std::make_shared<Function>((*Dmat)[6]);
                auto D1222_ptr = std::make_shared<Function>((*Dmat)[7]);
                auto D1212_ptr = std::make_shared<Function>((*Dmat)[8]);

                D1111_ptr->vector()->get_local(D1111_vec);
                D1122_ptr->vector()->get_local(D1122_vec);
                D1112_ptr->vector()->get_local(D1112_vec);
                D2211_ptr->vector()->get_local(D2211_vec);
                D2222_ptr->vector()->get_local(D2222_vec);
                D2212_ptr->vector()->get_local(D2212_vec);
                D1211_ptr->vector()->get_local(D1211_vec);
                D1222_ptr->vector()->get_local(D1222_vec);
                D1212_ptr->vector()->get_local(D1212_vec);

                for (int j = 0; j < dof_loc; j++){ //store values to vector 
                    Dmat1111_stzs_vec[ j + stz_addnew * dof_loc ] = D1111_vec[j];
                    Dmat1122_stzs_vec[ j + stz_addnew * dof_loc ] = D1122_vec[j];
                    Dmat1112_stzs_vec[ j + stz_addnew * dof_loc ] = D1112_vec[j];
                    Dmat2211_stzs_vec[ j + stz_addnew * dof_loc ] = D2211_vec[j];
                    Dmat2222_stzs_vec[ j + stz_addnew * dof_loc ] = D2222_vec[j];
                    Dmat2212_stzs_vec[ j + stz_addnew * dof_loc ] = D2212_vec[j];
                    Dmat1211_stzs_vec[ j + stz_addnew * dof_loc ] = D1211_vec[j];
                    Dmat1222_stzs_vec[ j + stz_addnew * dof_loc ] = D1222_vec[j];
                    Dmat1212_stzs_vec[ j + stz_addnew * dof_loc ] = D1212_vec[j];
                }

                std::vector<double> H111_vec, H122_vec, H112_vec, H211_vec, H222_vec, H212_vec;

                auto Hclass = std::make_shared<tensor_H>(stz_add_coords);
                *Hmat = *Hclass;

                auto H111_ptr = std::make_shared<Function>((*Hmat)[0]);
                auto H122_ptr = std::make_shared<Function>((*Hmat)[1]);
                auto H112_ptr = std::make_shared<Function>((*Hmat)[2]);
                auto H211_ptr = std::make_shared<Function>((*Hmat)[3]);
                auto H222_ptr = std::make_shared<Function>((*Hmat)[4]);
                auto H212_ptr = std::make_shared<Function>((*Hmat)[5]);

                H111_ptr->vector()->get_local(H111_vec);
                H122_ptr->vector()->get_local(H122_vec);
                H112_ptr->vector()->get_local(H112_vec);
                H211_ptr->vector()->get_local(H211_vec);
                H222_ptr->vector()->get_local(H222_vec);
                H212_ptr->vector()->get_local(H212_vec);

                for (int j = 0; j < dof_loc; j++){ //store values to vector 
                    Hmat111_stzs_vec[ j + stz_addnew * dof_loc ] = H111_vec[j];
                    Hmat122_stzs_vec[ j + stz_addnew * dof_loc ] = H122_vec[j];
                    Hmat112_stzs_vec[ j + stz_addnew * dof_loc ] = H112_vec[j];
                    Hmat211_stzs_vec[ j + stz_addnew * dof_loc ] = H211_vec[j];
                    Hmat222_stzs_vec[ j + stz_addnew * dof_loc ] = H222_vec[j];
                    Hmat212_stzs_vec[ j + stz_addnew * dof_loc ] = H212_vec[j];
                }

            }

        }

        //###############################################//
        int need_step = std::round(t_d / dt_current);
        
        //local processor loops its own stzs
        for (int stz_iter : locs_ownership){

            //check reactivation status
            if ( reactivation == true ){
                auto react_status = reactivate_indicator[stz_iter];
                if ( react_status > 0.0 ){
                    reactivate_indicator[stz_iter] -= 1; 
                }
                if ( reactivate_indicator[stz_iter] == 0 ){
                    active_status[stz_iter] = 0; //in the plot this should be differeniated from not active stz (finish)
                }
            }
            
            //Compute shear strain energy at each stz site (stzs) ->engs
            //!May need to modify here to include volumertic change!
            stz_class->strain_energy_local(phi_engs, locs, engs, stz_iter); //eval

            if (engs[stz_iter] > engs_c[stz_iter] && active_status[stz_iter] != 3 && active_status[stz_iter] != 1){ //activate change to 1 once it is activated and not finishing activate
                active_status[stz_iter] = -2;
                if (reactivate_indicator[stz_iter] >= -2 * dt_current && reactivate_indicator[stz_iter] <= 0.0){
                    std::cout<<"! STZ REACTIVATED ! -> "<<stz_iter<<std::endl;
                }
            }

            if (active_status[stz_iter] == 1 || active_status[stz_iter] == -2){

                //Store activated stz index 
                activated_stz_vec.push_back(stz_iter);
                
                //once activates, the angle remains constant
                if ( active_status[stz_iter] == -2 ){
                    //Compute local maximum shear direction vector s & n (current stz) //eval ->svec nvec
                    stz_class->coord_sn_local(sts_total, svec, nvec, locs, stz_iter);
                    active_status[stz_iter] = 1;
                }

                //Compute configurational force (current stz) //eval ->configf
                //stz_class->config_force_local(sts_image, sts_tilde, svec, nvec, configf, locs, stz_iter);
                stz_class->config_force_local_test(sts_total, svec, nvec, configf, locs, stz_iter);

                //Volumetric part
                stz_class->config_force_local_biaxial(sts_total, svec, nvec, configf_v, locs, stz_iter); 

                //Compute rate of shear strain (current stz) ->gamma_vec, active_status (1->possible->2), gamma_step_vec
                stz_class->shear_strain_rate_local(configf, gammadot, stz_iter, gamma_vec, active_status, gamma_step_vec, option, dt_current, reactivate_indicator); 

                //Volumetric part
                stz_class->biaxial_strain_rate_local(configf_v, epsilondot, stz_iter, epsilon_vec, active_status, epsilon_step_vec, option, dt_current, reactivate_indicator);

                //Compute rate of eigenstrain (current stz) ->eigen_eps -- eigen_eps_mpi
                stz_class->eigen_strain(gamma_step_vec, svec, nvec, eigen_eps, stz_iter);

                //Volumetric part
                stz_class->eigen_strain_volumetric(epsilon_step_vec, svec, nvec, eigen_eps_v, stz_iter);

                //Compute new potential stzs sites (4 around current one) -> new_placed_stz_coord -- new_placed_stz_coord_global
                //!Probably need to modify for volumetric case!
                stz_class->new_stz_coord(locs, svec, nvec, stz_iter, new_placed_stz_coord);

            }
        
        }

        //------------------MPI Collective Op START--------------------------//
        //------------------MPI Collective Op START--------------------------//

        //**active status** //mpi vec only for storing/plotting, modify vec in local
        MPI_Allreduce(active_status.data(), active_status_mpi.data(), active_status_mpi.size(), MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        //**reactive_indicator** //mpi vec only for storing/plotting, modify vec in local
        MPI_Allreduce(reactivate_indicator.data(), reactivate_indicator_mpi.data(), reactivate_indicator_mpi.size(), MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        //**eigen_eps** //mpi vec only for storing/plotting, modify vec in local
        MPI_Allreduce(eigen_eps.data(), eigen_eps_mpi.data(), eigen_eps_mpi.size(), MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        //Volumetric part -> eigen_eps_v
        MPI_Allreduce(eigen_eps_v.data(), eigen_eps_mpiv.data(), eigen_eps_mpiv.size(), MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        //**gamma_vec** //mpi vec only for storing/plotting, modify vec in local
        MPI_Allreduce(gamma_vec.data(), gamma_vec_mpi.data(), gamma_vec_mpi.size(), MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        //Volumetric part -> epsilon_vec_mpi
        MPI_Allreduce(epsilon_vec.data(), epsilon_vec_mpi.data(), epsilon_vec_mpi.size(), MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        //**new_placed_stz_coord**
        //Get local new stz vec size
        new_stz_count_local = new_placed_stz_coord.size(); 

        //Perform Allgather Op on the new stz number
        MPI_Allgather(&new_stz_count_local,1,MPI_INT,new_stz_count_global.data(),1,MPI_INT,MPI_COMM_WORLD); //twice the number of stzs

        //Compute displs buffer
        new_stz_displs[0] = 0.0;
        for (int m = 1; m < MPI::size(MPI_COMM_WORLD); m++){
            new_stz_displs[m] = new_stz_displs[m-1] + new_stz_count_global[m-1]; 
        }

        //Total number of new stzs elements
        int new_stz_total = std::accumulate(std::begin(new_stz_count_global), std::end(new_stz_count_global), 0);
        
        //Initialize receive buffer
        std::vector<double> new_placed_stz_coord_global(new_stz_total, 0.0); //array for stroing newly placed stzs (for global)

        //Perform Allgatherv Op on new stz coordinate
        MPI_Allgatherv(new_placed_stz_coord.data(),new_placed_stz_coord.size(),MPI_DOUBLE,new_placed_stz_coord_global.data(),new_stz_count_global.data(),new_stz_displs.data(),MPI_DOUBLE,MPI_COMM_WORLD);
        
        //------------------MPI Collective Op END----------------------------//
        //------------------MPI Collective Op END----------------------------//

        //Combine volumetric + shear contribution to eigen_eps_mpi
        //Note: eigen_eps_mpi should be re-computed at each time when doing mpi 
        for ( int i = 0; i < eigen_eps_mpi.size(); i++ ){
            eigen_eps_mpi[i] = eigen_eps_mpi[i] + eigen_eps_mpiv[i];
        }

        //output info
        int notactnum   = std::count(active_status_mpi.begin(),active_status_mpi.end(), 0);
        int curactnum   = std::count(active_status_mpi.begin(),active_status_mpi.end(), 1);
        int theactnum   = std::count(active_status_mpi.begin(),active_status_mpi.end(), 2);
        int endactnum   = std::count(active_status_mpi.begin(),active_status_mpi.end(), 3);
        if (MPI::rank(MPI_COMM_WORLD) == 0){
            std::cout<<"------------------------------------"<<std::endl;
            std::cout<<" STZ Num:                    "<<N_new<<std::endl;
            std::cout<<"------------------------------------"<<std::endl;
            std::cout<<"   Not    Activated STZ Num: "<<notactnum<<std::endl;
            std::cout<<"------------------------------------"<<std::endl;
            std::cout<<" Current  Activated STZ Num: "<<curactnum<<std::endl; 
            std::cout<<"Threshold Activated STZ Num: "<<theactnum<<std::endl; 
            std::cout<<"   End    Activated STZ Num: "<<endactnum<<std::endl; 
            std::cout<<"------------------------------------"<<std::endl;
        }

        //save mid_data
        if ( MPI::rank(MPI_COMM_WORLD) == 0 && step_num % 100 == 0 ){
            mid_data.open ("data/mid_data"+std::to_string(step_num)+".txt");
            for (int i = 0; i < locs.size() / 2; i++){
                mid_data <<(locs[0 + i * 2]) << " " <<(locs[1 + i * 2])<< " " << active_status_mpi[i] << std::endl;
            }
            mid_data.close();
        }

        //save mid_reactive
        if ( MPI::rank(MPI_COMM_WORLD) == 0 && reactivation == true ){
            mid_reactive.open ("curve/mid_reactive.txt");
            for ( int stz_index = 0; stz_index < reactivate_indicator_mpi.size(); stz_index++ ){
                mid_reactive<<reactivate_indicator_mpi[stz_index]<<" "<<active_status_mpi[stz_index]<<" "<<gamma_vec_mpi[stz_index]<<std::endl;
            }
            mid_reactive.close();
        }

        //Initialization
        double stz_loc[2];                    //stz location
        eps_tilde_step->vector()->zero(); //local strain tilde function
        u_tilde_step->vector()->zero();   //local displs tilde function
        sts_tilde_step->vector()->zero(); //local stress tilde function

        std::vector<double> eps11t_vec(dof_loc, 0.0), eps12t_vec(dof_loc, 0.0), eps21t_vec(dof_loc, 0.0), eps22t_vec(dof_loc, 0.0); //tilde strain (apply summation)
        std::vector<double> u1t_vec(dof_loc, 0.0), u2t_vec(dof_loc, 0.0);                                                           //tilde displs (apply summation)
        std::vector<double> sts11t_vec(dof_loc,0.0), sts12t_vec(dof_loc,0.0), sts21t_vec(dof_loc,0.0), sts22t_vec(dof_loc,0.0);     //tilde stress (apply summation)

        std::vector<double> eps11stz_vec(dof_loc, 0.0), eps12stz_vec(dof_loc, 0.0), eps21stz_vec(dof_loc, 0.0), eps22stz_vec(dof_loc, 0.0);

        for (int stz_iter = 0; stz_iter < active_status_mpi.size(); stz_iter++){
            
            //if globally one stz is 1 or 2, all local processors needs to update data of its portion, so we need mpi vec
            if (active_status_mpi[stz_iter] != 0 && active_status_mpi[stz_iter] != 3) //if the stz is still activated ( 2 will be placed by 3 once the loop finishes )
            {

                auto eigeneps11_iter = eigen_eps_mpi[0 + stz_iter * 4]; //the eigenstrain for that stz needs to be global
                auto eigeneps12_iter = eigen_eps_mpi[1 + stz_iter * 4];
                auto eigeneps21_iter = eigen_eps_mpi[2 + stz_iter * 4];
                auto eigeneps22_iter = eigen_eps_mpi[3 + stz_iter * 4];

                //-------------------------Update tilde strain-------------------------//

                for (int j = 0; j < dof_loc; j++){
                    
                    //current stz eps
                    eps11stz_vec[j] = Dmat1111_stzs_vec[ j + stz_iter * dof_loc ] * eigeneps11_iter + Dmat1112_stzs_vec[ j + stz_iter * dof_loc ] * eigeneps12_iter + Dmat1112_stzs_vec[ j + stz_iter * dof_loc ] * eigeneps21_iter + Dmat1122_stzs_vec[ j + stz_iter * dof_loc ] * eigeneps22_iter;
                    eps12stz_vec[j] = Dmat1211_stzs_vec[ j + stz_iter * dof_loc ] * eigeneps11_iter + Dmat1212_stzs_vec[ j + stz_iter * dof_loc ] * eigeneps12_iter + Dmat1212_stzs_vec[ j + stz_iter * dof_loc ] * eigeneps21_iter + Dmat1222_stzs_vec[ j + stz_iter * dof_loc ] * eigeneps22_iter;
                    eps21stz_vec[j] = Dmat1211_stzs_vec[ j + stz_iter * dof_loc ] * eigeneps11_iter + Dmat1212_stzs_vec[ j + stz_iter * dof_loc ] * eigeneps12_iter + Dmat1212_stzs_vec[ j + stz_iter * dof_loc ] * eigeneps21_iter + Dmat1222_stzs_vec[ j + stz_iter * dof_loc ] * eigeneps22_iter;
                    eps22stz_vec[j] = Dmat2211_stzs_vec[ j + stz_iter * dof_loc ] * eigeneps11_iter + Dmat2212_stzs_vec[ j + stz_iter * dof_loc ] * eigeneps12_iter + Dmat2212_stzs_vec[ j + stz_iter * dof_loc ] * eigeneps21_iter + Dmat2222_stzs_vec[ j + stz_iter * dof_loc ] * eigeneps22_iter;
                    
                    //summation stz eps
                    eps11t_vec[j] += Dmat1111_stzs_vec[ j + stz_iter * dof_loc ] * eigeneps11_iter + Dmat1112_stzs_vec[ j + stz_iter * dof_loc ] * eigeneps12_iter + Dmat1112_stzs_vec[ j + stz_iter * dof_loc ] * eigeneps21_iter + Dmat1122_stzs_vec[ j + stz_iter * dof_loc ] * eigeneps22_iter;
                    eps12t_vec[j] += Dmat1211_stzs_vec[ j + stz_iter * dof_loc ] * eigeneps11_iter + Dmat1212_stzs_vec[ j + stz_iter * dof_loc ] * eigeneps12_iter + Dmat1212_stzs_vec[ j + stz_iter * dof_loc ] * eigeneps21_iter + Dmat1222_stzs_vec[ j + stz_iter * dof_loc ] * eigeneps22_iter;
                    eps21t_vec[j] += Dmat1211_stzs_vec[ j + stz_iter * dof_loc ] * eigeneps11_iter + Dmat1212_stzs_vec[ j + stz_iter * dof_loc ] * eigeneps12_iter + Dmat1212_stzs_vec[ j + stz_iter * dof_loc ] * eigeneps21_iter + Dmat1222_stzs_vec[ j + stz_iter * dof_loc ] * eigeneps22_iter;
                    eps22t_vec[j] += Dmat2211_stzs_vec[ j + stz_iter * dof_loc ] * eigeneps11_iter + Dmat2212_stzs_vec[ j + stz_iter * dof_loc ] * eigeneps12_iter + Dmat2212_stzs_vec[ j + stz_iter * dof_loc ] * eigeneps21_iter + Dmat2222_stzs_vec[ j + stz_iter * dof_loc ] * eigeneps22_iter;
                
                }

                //-------------------------Update tilde displs-------------------------//

                for (int j = 0; j < dof_loc; j++){
                
                    u1t_vec[j] += Hmat111_stzs_vec[ j + stz_iter * dof_loc ] * eigeneps11_iter + Hmat112_stzs_vec[ j + stz_iter * dof_loc ] * eigeneps12_iter + Hmat112_stzs_vec[ j + stz_iter * dof_loc ] * eigeneps21_iter + Hmat122_stzs_vec[ j + stz_iter * dof_loc ] * eigeneps22_iter;
                    u2t_vec[j] += Hmat211_stzs_vec[ j + stz_iter * dof_loc ] * eigeneps11_iter + Hmat212_stzs_vec[ j + stz_iter * dof_loc ] * eigeneps12_iter + Hmat212_stzs_vec[ j + stz_iter * dof_loc ] * eigeneps21_iter + Hmat222_stzs_vec[ j + stz_iter * dof_loc ] * eigeneps22_iter;
                
                }

                //-------------------------Update tilde stress-------------------------//

                std::vector<double> epst_tr_vec(dof_loc,0.0);

                //Compute tr(strain)
                for (int i = 0; i < dof_loc; i++){ //loop over number of ptrs
                    epst_tr_vec[i] = eps11stz_vec[i] + eps22stz_vec[i]; 
                }

                for (int j = 0; j < dof_loc; j++){ //loop over number of ptrs
                    sts11t_vec[j] += E / ( 1 + nu ) * ( eps11stz_vec[j] + nu / ( 1 - 2 * nu ) * epst_tr_vec[j] );
                    sts22t_vec[j] += E / ( 1 + nu ) * ( eps22stz_vec[j] + nu / ( 1 - 2 * nu ) * epst_tr_vec[j] );
                    sts12t_vec[j] += E / ( 1 + nu ) * ( eps12stz_vec[j]                                        );
                    sts21t_vec[j] += E / ( 1 + nu ) * ( eps21stz_vec[j]                                        );
                }

                //set threshold to end situation
                if ( active_status[stz_iter] == 2 ){ active_status[stz_iter] = 3;} //update local active_status //only the owner should be 2, others are zero in the same positon

                //update reactivation
                if ( reactivation == true ){
                    if ( active_status[stz_iter] == 3 && reactivate_indicator[stz_iter] <= 0 ){
                        reactivate_indicator[stz_iter] = need_step;
                    }
                }
            }
        
        }

        //Assign step tilde field to function
        //------------------------eps_tilde_step-----------------------------//
        //Get tilde strain component pointer 
        auto eps11stz_ptr = std::make_shared<Function>((*eps_tilde_step)[0]);
        auto eps12stz_ptr = std::make_shared<Function>((*eps_tilde_step)[1]);
        auto eps21stz_ptr = std::make_shared<Function>((*eps_tilde_step)[2]);
        auto eps22stz_ptr = std::make_shared<Function>((*eps_tilde_step)[3]);

        //Set strain component vectors back to function
        eps11stz_ptr->vector()->set_local(eps11t_vec);
        eps12stz_ptr->vector()->set_local(eps12t_vec);
        eps21stz_ptr->vector()->set_local(eps21t_vec);
        eps22stz_ptr->vector()->set_local(eps22t_vec);

        eps11stz_ptr->vector()->apply("insert");
        eps12stz_ptr->vector()->apply("insert");
        eps21stz_ptr->vector()->apply("insert");
        eps22stz_ptr->vector()->apply("insert");

        assign(eps_tilde_step,{eps11stz_ptr,eps12stz_ptr,eps21stz_ptr,eps22stz_ptr});

        //------------------------u_tilde_step------------------------------//
        auto u1stz_ptr = std::make_shared<Function>((*u_tilde_step)[0]);
        auto u2stz_ptr = std::make_shared<Function>((*u_tilde_step)[1]);

        u1stz_ptr->vector()->set_local(u1t_vec);
        u2stz_ptr->vector()->set_local(u2t_vec);

        u1stz_ptr->vector()->apply("insert");
        u2stz_ptr->vector()->apply("insert");

        assign(u_tilde_step,{u1stz_ptr,u2stz_ptr});

        //------------------------sts_tilde_step----------------------------//

        //Get stress components
        auto sts11stz_ptr = std::make_shared<Function>((*sts_tilde_step)[0]);
        auto sts12stz_ptr = std::make_shared<Function>((*sts_tilde_step)[1]);
        auto sts21stz_ptr = std::make_shared<Function>((*sts_tilde_step)[2]);
        auto sts22stz_ptr = std::make_shared<Function>((*sts_tilde_step)[3]);

        //Set stress components back into stress function
        sts11stz_ptr->vector()->set_local(sts11t_vec);
        sts12stz_ptr->vector()->set_local(sts12t_vec);
        sts21stz_ptr->vector()->set_local(sts21t_vec);
        sts22stz_ptr->vector()->set_local(sts22t_vec);
        
        sts11stz_ptr->vector()->apply("insert");
        sts12stz_ptr->vector()->apply("insert");
        sts21stz_ptr->vector()->apply("insert");
        sts22stz_ptr->vector()->apply("insert");

        assign(sts_tilde_step,{sts11stz_ptr,sts12stz_ptr,sts21stz_ptr,sts22stz_ptr});

        //------------------------------------------------------------------//

        //Update new stzs in locs -> locs, N0
        N0 = N_new;
        stz_class->update_new_stz(new_placed_stz_coord_global, locs);
        N_new = locs.size() / 2;

        new_stz_flag = false;
        if (N_new > N0){ new_stz_flag = true; }

        //Add id3 tilde strain field
        eps_tilde->vector()->axpy(1.0, *eps_tilde_step->vector());
        u_tilde->vector()->axpy(1.0, *u_tilde_step->vector());
        sts_tilde->vector()->axpy(1.0, *sts_tilde_step->vector());

        //Update uniaxial tension boundary //simple shear boundary
        //simple_shear_boundary->_tcur = t_cur;
        if ( dt_flag == 1 ){
            //simple_shear_boundary->_tcur = dt; //increment
            axial_tension->_tcur = dt; //increment
        }
        else{
            //simple_shear_boundary->_tcur = dt_small;
            axial_tension->_tcur = dt_small;
        }

        //Solve IBVP for image field displacement
        ns_vf.solve(nseq_eps_vf, *u_image_step->vector());

        //Project for image field strain
        ns_project.solve(nseq_eps_project, *eps_image_step->vector());

        //Add increment to total image field
        u_image->vector()->axpy(1.0, *u_image_step->vector());
        eps_image->vector()->axpy(1.0, *eps_image_step->vector());

        //Compute image field stress (->sts_image)
        stz_class->image_stress(eps_image, sts_image);

        //Compute total field strain (->eps_total)
        stz_class->total_strain(eps_image, eps_tilde, eps_total);

        //Compute total field stress (->sts_total)
        stz_class->total_stress(sts_image, sts_tilde, sts_total);

        //Compute total field displs (->u_total)
        stz_class->total_displs(u_image, u_tilde, u_total);

        //Save pvd files
        if (step_num % 1 == 0){
            
            /*
            File epsfile("res/straint/eps"+std::to_string(step_num)+".pvd");
            epsfile << *eps_total;
            
            File stsfile("res/stresst/sts"+std::to_string(step_num)+".pvd");
            stsfile << *sts_total;

            File ufile("res/displst/uu"+std::to_string(step_num)+".pvd");
            ufile << *u_total;

            File phifile("res/energy/phi"+std::to_string(step_num)+".pvd");
            phifile << *phi_engs;
            */

        }
        {
            auto s12 = std::make_shared<sts_boundary::Form_sts_bc>(mesh,sts_total);
            s12->ds = boundaries;
            double str12bc = assemble(*s12) / L;

            if (MPI::rank(MPI_COMM_WORLD) == 0){
                mid_curve.open ("curve/mid_curve.txt",std::ios::app);
                    mid_curve << std::to_string(gamma_dot*t_cur) + " " + std::to_string(str12bc) + "\n";
                mid_curve.close();
            }

        }

    }
 
    if (MPI::rank(MPI_COMM_WORLD) == 0){
        for (int i = 0; i < locs.size() / 2; i++){
            stzlocs << std::to_string(locs[0 + i * 2])<<" "<<std::to_string(locs[1 + i * 2]) + "\n";     
        }
        for (int j = 0; j < active_status.size(); j++){
            actindx << std::to_string(active_status[j]) << "\n";
        }
    }
        
}
