"""Author: Vinut Raibagi
   email : vinutr11@gmail.com"""

'''Code to solve crack propagation for single edge notched composite lamina'''

from fenics import *
import numpy as np
import matplotlib.pyplot as plt
from csv import DictWriter
import time as Time
from ufl import max_value
import os, mshr

tic = Time.time()

parameters['form_compiler']['cpp_optimize'] = True
parameters['form_compiler']['optimize'] = True
set_log_active(False)

# Creating mesh and defining functionspace
rectangle = mshr.Rectangle(Point(0,0),Point(1,2))
crack = mshr.Rectangle(Point(0,0.995),Point(0.5,1.005))
domain = rectangle - crack

mesh = mshr.generate_mesh(domain,200)
plot(mesh)
plt.show()

dim = mesh.geometry().dim()
hmin = mesh.hmin()
print('Minimum cell size is ',hmin)
print('Number of cells are ',mesh.num_cells())

V = FunctionSpace(mesh,"CG",1)                  # For phase field
W = VectorFunctionSpace(mesh,"CG",1)            # For displacement
WW = FunctionSpace(mesh,"DG",0)                 # For projecting Energy
TT = TensorFunctionSpace(mesh,'DG',0,(2,2))     # For projecting stress and strains 

print(f'Number of degrees of freedom for phase field {V.dim()}')
print(f'Number of degrees of freedom for displacement {W.dim()}')

# Trial and test functions
p , q = TrialFunction(V) , TestFunction(V)
u , v = TrialFunction(W) , TestFunction(W)

# Material properties
E11 = Constant(114.8e3)                                                          # Longitudinal stiffness
E22 = Constant(11.7e3)                                                           # Transverse stiffness
G12 = Constant(9.66e3)                                                           # Shear stiffness
nu12 = Constant(0.21)                                                            # Poisson's ratio
Gf = Constant(106.3)                                                             # Longitudinal fracture toughness
Gm1 = Constant(0.2774)                                                           # Transverse normal fracture toughness
Gm2 = Constant(0.7879)                                                           # Transverse shear fracture toughness

l_0 = Constant(0.01)                                                             # Length scale parameter
beta = Constant(15)                                                              # MPP

alpha = 30                                                                       # Angle in degrees
t = (alpha*pi)/180                                                               

N = as_vector([cos(t), sin(t)])
Omega = Identity(dim) + beta*outer(N,N)

S = as_matrix([[1/E11, -nu12/E11, 0.],[-nu12/E11, 1/E22, 0.],[0., 0., 1/(2*G12)]])       # Compliance matrix
C = inv(S)                                                                               # Stiffness matrix

T = as_matrix([[cos(t)**2, sin(t)**2, -2*cos(t)*sin(t)],[sin(t)**2, cos(t)**2, 2*cos(t)*sin(t)],[cos(t)*sin(t), -cos(t)*sin(t), cos(t)**2-sin(t)**2]])                                                                                   # Transformation matrix
T_inv = inv(T)                                                                           # Inverse of T

# Subdomains/Boundaries
bottom = CompiledSubDomain('on_boundary and near(x[1],0)')
top = CompiledSubDomain('on_boundary and near(x[1],2)')

boundaries = MeshFunction('size_t',mesh,mesh.topology().dim()-1)
boundaries.set_all(0)
bottom.mark(boundaries,1)
ds = Measure('ds',subdomain_data=boundaries)

n = FacetNormal(mesh)

load = Expression('t',t=0,degree=1)

bc_bottom = DirichletBC(W,Constant((0,0)),bottom)
bc_top = DirichletBC(W.sub(1),load,top)
bcs = [bc_bottom, bc_top]

# Definning fields
def epsilon(u):
    '''Strain in global coordinate system'''
    return sym(grad(u))

def tensor_to_voigt(tensor):
    '''Function to convert tensor into voigt(vector)'''
    return as_vector([tensor[0,0], tensor[1,1], tensor[0,1]])

def voigt_to_tensor(voigt):
    '''Function to convert voigt(vector) into tensor'''
    return as_tensor([[voigt[0], voigt[2]],[voigt[2], voigt[1]]])

def sigma(u):
    '''Constitutive relation for stress and strain for orthotropic linear elastic model under plane stress condition. Stress in global coordinate system '''
    strain = epsilon(u)
    strain_to_voigt = tensor_to_voigt(strain)
    Q = T*C*T_inv                                                             # Global stiffness matrix
    voigt_stress = dot(Q, strain_to_voigt)
    return voigt_to_tensor(voigt_stress)

def Df(u):
    '''Crack driving force'''
    strain_mat, stress_mat = dot(T_inv,tensor_to_voigt(epsilon(u))), dot(T_inv,tensor_to_voigt(sigma(u)))
    psi_fiber = Constant(0.5)*((stress_mat[0] + abs(stress_mat[0]))/2)*((strain_mat[0] + abs(strain_mat[0]))/2)
    psi_matrix_1 = Constant(0.5)*((stress_mat[1] + abs(stress_mat[1]))/2)*((strain_mat[1] + abs(strain_mat[1]))/2)
    psi_matrix_2 = stress_mat[2]*strain_mat[2]
    return ((psi_fiber*sqrt(1+beta))/Gf) + (psi_matrix_1/Gm1) + (psi_matrix_2/Gm2)

def H_project(u,WW,H):
    '''Update the History field'''
    WW_trial = TrialFunction(WW)
    WW_test = TestFunction(WW)

    psi_max = max_value(Df(u),H)
    a_proj = inner(WW_trial,WW_test)*dx
    b_proj = inner(psi_max,WW_test)*dx

    solver = LocalSolver(a_proj,b_proj)
    solver.factorize()
    solver.solve_global_rhs(H)

u_new = Function(W, name = 'Displacement u_(n+1)')     
u_old = Function(W, name = 'Displacement u_(n)')       
pnew = Function(V, name = 'Phase_field p_(n+1)')
pold = Function(V, name = 'Phase_field p_(n)')
H = Function(WW, name = 'History_field')

# Equilibrium equation
form_u = ((1 - pold)**2)*inner(sigma(u),grad(v))*dx

# Phase field form equation
form_phi = (((p*q)/l_0) + l_0*dot(grad(q),dot(Omega,grad(p))))*dx - (2*(1-p)*H*q)*dx

disp_problem = LinearVariationalProblem(lhs(form_u),rhs(form_u),u_new,bcs=bcs)
phi_problem = LinearVariationalProblem(lhs(form_phi),rhs(form_phi),pnew)

disp_solver = LinearVariationalSolver(disp_problem)
phi_solver = LinearVariationalSolver(phi_problem)

u_max = 0.025
time = np.linspace(0,u_max)
size = time.size
Reaction = np.zeros(size)

tol = 5e-4

folder_name = 'Composite_results_SEN'
os.mkdir(folder_name)

file = open(os.path.join(folder_name,'Results.csv'),'w')  
csv_wr = DictWriter(file,fieldnames=['Displacement','Load'])
csv_wr.writeheader()

xdmf_file = XDMFFile(os.path.join(folder_name,'Results.xdmf'))
xdmf_file.parameters['flush_output'] = True
xdmf_file.parameters['functions_share_mesh'] = True
xdmf_file.parameters['rewrite_function_mesh'] = False

# Staggered scheme
for (i,t) in enumerate(time):
    load.t = t                        # Updating load
    iter = 0
    err = 1

    while err > tol:
        iter += 1
        disp_solver.solve()           # Solving equilibrium equation
        H_project(u_new,WW,H)         # Updating history field
        phi_solver.solve()            # Solving phase field equation

        err_u = errornorm(u_new,u_old,'l2')
        err_phi = errornorm(pnew,pold,'l2')
        err = max(err_u,err_phi)

        u_old.assign(u_new)
        pold.assign(pnew)

        if err < tol:
            print(f'Displacement : {t} mm, iterations : {iter}')
            Traction = dot(((1-pnew)**2)*sigma(u_new),n)
            fy = assemble(Traction[1]*ds(1))
            Reaction[i] = fy

            if np.any(np.isnan(Reaction)):
                raise ValueError('Simulation stopped. nan is encountered in output')

            csv_wr.writerow({'Displacement':t,'Load':fy})

            xdmf_file.write(u_new,t)
            xdmf_file.write(pnew,t)
            
file.close()

plt.figure()
p1 = plot(pnew)
plt.colorbar(p1)
plt.savefig(os.path.join(folder_name,'Phase_field'))

plt.figure()
plt.tight_layout()
plt.plot(time, np.abs(Reaction))
plt.xlabel(r"Displacement [$u(mm)$]")
plt.ylabel(r"Load [$N$]")
plt.savefig(os.path.join(folder_name,'Load_displacement'))

toc = Time.time()

print('Total simulation time in minutes ',(toc-tic)/60)
plt.show()






