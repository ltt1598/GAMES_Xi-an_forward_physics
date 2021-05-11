import taichi as ti
import math

ti.init(arch=ti.cpu)

# global control
paused = ti.field(ti.i32, ())
damping_toggle = ti.field(ti.i32, ())
curser = ti.Vector.field(2, ti.f32, ())
picking = ti.field(ti.i32,())

# integration method
# 1: explicit euler
# 2: symplectic euler
# 3: (semi-)implicit euler
integration = 2

# procedurally setting up the cantilever
init_x, init_y = 0.1, 0.6
N_x = 20
N_y = 4
# N_x = 2
# N_y = 2
N = N_x*N_y
N_edges = (N_x-1)*N_y + N_x*(N_y - 1) + (N_x-1) * \
    (N_y-1)  # horizontal + vertical + diagonal springs
N_triangles = 2 * (N_x-1) * (N_y-1)
dx = 1/32
curser_radius = dx/2

# physical quantities
m = 1
g = 9.8
YoungsModulus = ti.field(ti.f32, ())

# time-step size (for simulation, 16.7ms)
h = 16.7e-3
# substepping
substepping = 100
# time-step size (for time integration)
dh = h/substepping

# simulation components
x = ti.Vector.field(2, ti.f32, N)
v = ti.Vector.field(2, ti.f32, N)
grad = ti.Vector.field(2, ti.f32, N)
Hess = ti.Matrix.field(2,2,ti.f32,(N,N)) # NxN dense matrix, ouch
spring_length = ti.field(ti.f32, N_edges)

# temp variables for cg
cg_b = ti.Vector.field(2, ti.f32, N)
cg_r = ti.Vector.field(2, ti.f32, N)
cg_d = ti.Vector.field(2, ti.f32, N)
cg_q = ti.Vector.field(2, ti.f32, N)
cg_A = ti.Matrix.field(2,2,ti.f32,(N,N)) 

# geometric components
triangles = ti.Vector.field(3, ti.i32, N_triangles)
edges = ti.Vector.field(2, ti.i32, N_edges)

def ij_2_index(i, j): return i * N_y + j

# -----------------------meshing and init----------------------------
@ti.kernel
def meshing():
    # setting up triangles
    for i in range(N_x - 1):
        for j in range(N_y - 1):
            # triangle id
            tid = (i * (N_y - 1) + j) * 2
            triangles[tid][0] = ij_2_index(i, j)
            triangles[tid][1] = ij_2_index(i + 1, j)
            triangles[tid][2] = ij_2_index(i, j + 1)

            tid = (i * (N_y - 1) + j) * 2 + 1
            triangles[tid][0] = ij_2_index(i, j + 1)
            triangles[tid][1] = ij_2_index(i + 1, j + 1)
            triangles[tid][2] = ij_2_index(i + 1, j)

    # setting up edges
    # edge id
    eid_base = 0

    # horizontal edges
    for i in range(N_x-1):
        for j in range(N_y):
            eid = eid_base+i*N_y+j
            edges[eid] = [ij_2_index(i, j), ij_2_index(i+1, j)]

    eid_base += (N_x-1)*N_y
    # vertical edges
    for i in range(N_x):
        for j in range(N_y-1):
            eid = eid_base+i*(N_y-1)+j
            edges[eid] = [ij_2_index(i, j), ij_2_index(i, j+1)]

    eid_base += N_x*(N_y-1)
    # diagonal edges
    for i in range(N_x-1):
        for j in range(N_y-1):
            eid = eid_base+i*(N_y-1)+j
            edges[eid] = [ij_2_index(i+1, j), ij_2_index(i, j+1)]


@ti.kernel
def initialize():
    YoungsModulus[None] = 3e7
    paused[None] = 1
    # init position and velocity
    for i in range(N_x):
        for j in range(N_y):
            index = ij_2_index(i, j)
            x[index] = [init_x + i * dx, init_y + j * dx]
            v[index].fill(0.0)

@ti.kernel
def initialize_springs():
    # init spring rest-length
    for i in range(N_edges):
        a, b = edges[i][0], edges[i][1]
        r = x[a]-x[b]
        spring_length[i] = r.norm()

# ----------------------core-----------------------------
@ti.func #Ax = A*x
def multiplication(Ax, A, x, n):
    for i in range(n):
        Ax[i].fill(0.0)
        for j in range(n):
            Ax[i] += A[i,j]@x[j]
@ti.func
def dot_product(a,b,n):
    res = 0.0
    for i in range(n):
        res += a[i].dot(b[i])
    return res

@ti.kernel
def compute_gradient():
    # clear gradient
    for i in range(N_edges):
        grad[i] = [0, 0]

    # gradient of elastic potential
    for i in range(N_edges):
        a, b = edges[i][0], edges[i][1]
        r = x[a]-x[b]
        l = r.norm()
        l0 = spring_length[i]
        k = YoungsModulus[None]*l0  # stiffness in Hooke's law
        gradient = k*(l-l0)*r/l
        grad[a] += gradient
        grad[b] += -gradient

@ti.kernel
def compute_hessian():
    #clear hessian
    for i in range(N_edges):
        for j in range(N_edges):
            Hess[i,j].fill(0.0)

    eye = ti.Matrix.cols([[1.0, 0.0], [0.0, 1.0]])
    # hessian of elastic potential
    for i in range(N_edges):
        a, b = edges[i][0], edges[i][1]
        r = x[a]-x[b]
        l = r.norm()
        l0 = spring_length[i]
        k = YoungsModulus[None]*l0  # stiffness in Hooke's law
        rrt = ti.Matrix.cols([[r.x*r.x, r.x*r.y], [r.y*r.x, r.y*r.y]])
        K = k*(eye-l0/l*(eye-rrt))
        Hess[a, a] += K
        Hess[a, b] -= K
        Hess[b, a] -= K
        Hess[b, b] += K

@ti.kernel
def update():
    # perform time integration
    if integration == 1:
        for i in range(N):
            # explicit euler integration
            x[i] += dh*v[i]   
            # elastic force + gravitation force, divding mass to get the acceleration
            acc = -grad[i]/m - [0.0, g]
            v[i] += dh*acc
    elif integration == 2:        
        for i in range(N):
            # symplectic integration
            # elastic force + gravitation force, divding mass to get the acceleration
            acc = -grad[i]/m - [0.0, g]
            v[i] += dh*acc
            x[i] += dh*v[i]
    elif integration == 3:
        #(semi-)implicit euler
        # v1 = v0 + h*M^-1*f
        for i in range(N):
            acc = -grad[i]/m - [0.0, g]
            v[i] += dh*acc

        # (M+h^2Hess) * v2 = M*v1
        # construct A and b, init v2 = 0
        for i in range(N):
            for j in range(N):
                cg_A[i,j]=Hess[i,j]*dh*dh
                if i==j:
                    cg_A[i,j][0,0] += m*1.0
                    cg_A[i,j][1,1] += m*1.0
            cg_b[i] = m*v[i]
            v[i].fill(0.0)

        # solve for A * v2 = b using conjugate gradient
        it, max_it = 0,100
        #r = b-Ax, d=r
        multiplication(cg_r, cg_A, v, N)
        for i in range(N):
            cg_r[i] = cg_b[i] - cg_r[i]
            cg_d[i] = cg_r[i]
        #delta = r^T * r
        delta_new = dot_product(cg_r, cg_r, N)
        delta_0 = delta_new 
        while it < max_it and delta_new/delta_0 > 1e-8:
            # q = A*d
            multiplication(cg_q, cg_A, cg_d, N)
            # alpha = delta / d^T A d
            alpha = delta_new / (dot_product(cg_d, cg_q, N))
            # v = v + alpha d, r = r - alpha q
            for i in range(N):
                v[i] += alpha * cg_d[i]
                cg_r[i] -= alpha * cg_q[i] # r = b-Ax or r = r - alpha r (equivalent if no numerical error)

            delta_old = delta_new 
            # delta_new = r^T * r
            delta_new = dot_product(cg_r, cg_r, N)
            beta = delta_new / delta_old
            # d = r + beta * d
            for i in range(N):
                cg_d[i] = cg_r[i] + beta * cg_d[i]
            it += 1

        # update position x+=h*v2
        for i in range(N):
            x[i] += dh*v[i]

    # explicit damping (ether drag)
    if damping_toggle:
        for i in range(N):
            v[i] *= ti.exp(-dh*5)

    # enforce boundary condition
    if picking[None]:
        for i in range(N):
            r = x[i]-curser[None]
            if r.norm() < curser_radius:
                x[i] = curser[None]
                v[i].fill(0.0)
                pass

    for j in range(N_y):
        ind = ij_2_index(0, j)
        v[ind] = [0, 0]
        x[ind] = [init_x, init_y + j * dx]  # rest pose attached to the wall

    for i in range(N):
        if x[i].x < init_x:
            x[i].x = init_x
            v[i].x = 0


# init once and for all
meshing()
initialize()
initialize_springs()

gui = ti.GUI('mass-spring system', (800, 800))
while gui.running:

    picking[None]=0

    # key events
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()
        elif e.key == 'r':
            initialize()
        elif e.key == '0':
            YoungsModulus[None] *= 1.1
        elif e.key == '9':
            YoungsModulus[None] /= 1.1
        elif e.key == ti.GUI.SPACE:
            paused[None] = not paused[None]
        elif e.key =='d' or e.key == 'D':
            damping_toggle[None] = not damping_toggle[None]
        elif e.key == 'p' or e.key == 'P':
            for i in range(substepping):
                compute_gradient()
                if integration == 3:
                    compute_hessian()
                update()           

    if gui.is_pressed(ti.GUI.LMB):
        curser[None] = gui.get_cursor_pos()
        picking[None] = 1

    # numerical time integration
    if not paused[None]:
        for i in range(substepping):
            compute_gradient()
            if integration == 3:
                compute_hessian()
            update()

    # render
    pos = x.to_numpy()
    for i in range(N_edges):
        a, b = edges[i][0], edges[i][1]
        gui.line((pos[a][0], pos[a][1]),
                 (pos[b][0], pos[b][1]),
                 radius=1,
                 color=0xFFFF00)
    gui.line((init_x, 0.0), (init_x, 1.0), color=0xFFFFFF, radius=4)

    if picking[None]:
        gui.circle((curser[None][0], curser[None][1]), radius=curser_radius*800, color=0xFF8888)

    # text
    gui.text(
        content=f'space: Paused: {paused[None]:d}', pos=(0.6, 0.925), color=0xFFFFFF)
    gui.text(
        content=f'9/0: (-/+) Young\'s Modulus {YoungsModulus[None]:.1f}', pos=(0.6, 0.9), color=0xFFFFFF)
    gui.text(
        content=f'D: Damping: {damping_toggle[None]:d}', pos=(0.6, 0.875), color=0xFFFFFF)
    gui.show()
