import taichi as ti
import math

ti.init(arch=ti.cpu)

# global control
paused = ti.field(ti.i32, ())
draw_grid = ti.field(ti.i32, ())
curser = ti.Vector.field(2, ti.f32, ())
picking = ti.field(ti.i32,())
damping_toggle = ti.field(ti.i32, ())

# procedurally setting up the cantilever
init_x, init_y = 0.1, 0.6
N_x = 32
N_y = 4
N = N_x*N_y
N_edges = (N_x-1)*N_y + N_x*(N_y - 1) + (N_x-1) * \
    (N_y-1)  # horizontal + vertical + diagonal springs
N_triangles = 2 * (N_x-1) * (N_y-1)
N_grid = 20
dx_meshing = 1/32
dx = 1 / N_grid
inv_dx = 1 / dx
curser_radius = dx_meshing/2

grid_buffer = 1

# physical quantities
m = 1
g = 9.8
YoungsModulus = ti.field(ti.f32, ())
PoissonsRatio = ti.field(ti.f32, ())
LameMu = ti.field(ti.f32, ())
LameLa = ti.field(ti.f32, ())

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
C = ti.Matrix.field(2, 2, ti.f32, N) # the affine transformation matrix in APIC
grid_v = ti.Vector.field(2, ti.f32, (N_grid, N_grid))
grid_m = ti.field(ti.f32, (N_grid, N_grid))
elements_Dm_inv = ti.Matrix.field(2, 2, ti.f32, N_triangles)
elements_V0 = ti.field(ti.f32, N_triangles)

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
    YoungsModulus[None] = 3e6
    PoissonsRatio[None] = 0.3
    # init position and velocity
    for i in range(N_x):
        for j in range(N_y):
            index = ij_2_index(i, j)
            x[index] = [init_x + i * dx_meshing, init_y + j * dx_meshing]
            v[index].fill(0)

@ti.func
def compute_D(i):
    a = triangles[i][0]
    b = triangles[i][1]
    c = triangles[i][2]
    return ti.Matrix.cols([x[b] - x[a], x[c] - x[a]])

@ti.kernel
def initialize_elements():
    for i in range(N_triangles):
        Dm = compute_D(i)
        elements_Dm_inv[i] = Dm.inverse()
        elements_V0[i] = ti.abs(Dm.determinant())/2

# ----------------------core-----------------------------
@ti.func
def compute_R_2D(F):
    R, S = ti.polar_decompose(F, ti.f32)
    return R

@ti.kernel
def compute_gradient():
    # clear gradient
    for i in range(N_edges):
        grad[i].fill(0)

    # gradient of elastic potential
    for i in range(N_triangles):
        Ds = compute_D(i)
        F = Ds@elements_Dm_inv[i]
        # co-rotated linear elasticity
        R = compute_R_2D(F)
        Eye = ti.Matrix.cols([[1.0, 0.0], [0.0, 1.0]])
        # first Piola-Kirchhoff tensor
        P = 2*LameMu[None]*(F-R) + LameLa[None]*((R.transpose())@F-Eye).trace()*R
        #assemble to gradient
        H = elements_V0[i] * P @ (elements_Dm_inv[i].transpose())
        a,b,c = triangles[i][0],triangles[i][1],triangles[i][2]
        gb = ti.Vector([H[0,0], H[1, 0]])
        gc = ti.Vector([H[0,1], H[1, 1]])
        ga = -gb-gc
        grad[a] += ga
        grad[b] += gb
        grad[c] += gc     

@ti.kernel
def p2g():
    for p in x:
        base = ti.cast(x[p] * inv_dx - 0.5, ti.i32) # floor
        fx = x[p] * inv_dx - ti.cast(base, float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2] # quadratic interpolation function
        affine = m * C[p]
        # 3x3 grid around that particle p
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                I = ti.Vector([i, j])
                dpos = (float(I) - fx) * dx
                weight = w[i].x * w[j].y
                grid_v[base + I] += weight * (m * v[p] - grad[p]*dh + affine @ dpos) #APIC
                grid_m[base + I] += weight * m

@ti.kernel
def g2p():
    for p in x:
        base = ti.cast(x[p] * inv_dx - 0.5, ti.i32)
        fx = x[p] * inv_dx - float(base)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector([0.0, 0.0])
        new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])

        # gather information back from 3x3 grid around
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                I = ti.Vector([i, j])
                dpos = float(I) - fx
                g_v = grid_v[base + I]
                weight = w[i].x * w[j].y
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx #APIC

        C[p] = new_C # affine transformation matrix for particle p

        # symplectic integration for particles
        v[p] = new_v
        x[p] += dh * v[p]

@ti.kernel
def grid_op():
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            inv_m = 1 / grid_m[i, j]
            grid_v[i, j] = inv_m * grid_v[i, j] # convert from momentum to velocity
            grid_v[i, j].y -= dh * g # advect with external force

    # particles picked by user
    if picking[None]:
        for p in range(N):
            r = x[p]-curser[None]
            if r.norm() < curser_radius:
                base = ti.cast(x[p] * inv_dx - 0.5, ti.i32)
                for i in ti.static(range(3)):
                    for j in ti.static(range(3)): 
                        I = ti.Vector([i, j])
                        grid_v[base + I] = -r / dh  

    if damping_toggle[None]:
        for i in range(N_grid):
            for j in range(N_grid):
                grid_v[i, j]*=ti.exp(-dh*4)

    # particles attached to the wall
    for jj in range(N_y):
        p_w = ij_2_index(0, jj)
        base = ti.cast(x[p_w] * inv_dx - 0.5, ti.i32)
        for i in ti.static(range(3)):
            for j in ti.static(range(3)): 
                I = ti.Vector([i, j])
                grid_v[base + I].fill(0)              

@ti.kernel
def updateLameCoeff():
    E = YoungsModulus[None]
    nu = PoissonsRatio[None]
    LameLa[None] = E*nu / ((1+nu)*(1-2*nu))
    LameMu[None] = E / (2*(1+nu))

# init once and for all
meshing()
initialize()
initialize_elements()
updateLameCoeff()

gui = ti.GUI('Linear FEM', (800, 800))
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
            if YoungsModulus[None] <= 0:
                YoungsModulus[None] = 0
        elif e.key == '8':
            PoissonsRatio[None] = PoissonsRatio[None]*0.9+0.05 # slowly converge to 0.5
            if PoissonsRatio[None] >= 0.499:
                PoissonsRatio[None] = 0.499
        elif e.key == '7':
            PoissonsRatio[None] = PoissonsRatio[None]*1.1-0.05
            if PoissonsRatio[None] <= 0:
                PoissonsRatio[None] = 0
        elif e.key == ti.GUI.SPACE:
            paused[None] = not paused[None]
        elif e.key =='g' or e.key == 'G':
            draw_grid[None] = not draw_grid[None]
        elif e.key =='d' or e.key == 'D':
            damping_toggle[None] = not damping_toggle[None]
        elif e.key =='p' or e.key == 'P': # step-forward
            for i in range(substepping):
                grid_m.fill(0)
            grid_v.fill(0)
            compute_gradient()
            p2g()
            grid_op()
            g2p()
        updateLameCoeff()

    if gui.is_pressed(ti.GUI.LMB):
        curser[None] = gui.get_cursor_pos()
        picking[None] = 1

    # numerical time integration
    if not paused[None]:
        for i in range(substepping):
            grid_m.fill(0)
            grid_v.fill(0)
            compute_gradient()
            p2g()
            grid_op()
            g2p()

    # show grid
    if draw_grid[None]:
        for i in range(N_grid):
            gui.line((0.0, i/N_grid),
                    (1.0, i/N_grid),
                    radius=1,
                    color=0x888888)
        for i in range(N_grid):
            gui.line((i/N_grid, 0.0),
                    (i/N_grid, 1.0),
                    radius=1,
                    color=0x888888)

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
        content=f'G:show/hide grid: {draw_grid[None]:d}', pos=(0.6, 0.925), color=0xFFFFFF)
    gui.text(
        content=f'9/0: (-/+) Young\'s Modulus {YoungsModulus[None]:.1f}', pos=(0.6, 0.9), color=0xFFFFFF)
    gui.text(
        content=f'7/8: (-/+) Poisson\'s Ratio {PoissonsRatio[None]:.3f}', pos=(0.6, 0.875), color=0xFFFFFF)
    gui.text(
        content=f'D: Damping: {damping_toggle[None]:d}', pos=(0.6, 0.85), color=0xFFFFFF)

    gui.show()
