# using DifferentialEquations # This package takes too much time to compile

# Define robot kinematic model
# function car_ode!(du, u, p, t)
#     du[1] = u[2]
#     du[2] = p
# end

# Dynamics: Kinematic model
function car_model(x, u, dt, p)
    θ = x[3]; v = x[4]
    a = u[1]; δ = u[2]
    lf = p[1]; lr = p[2]
    β = atan(lr/(lf+lr)*tan(δ))
    dx = zeros(4)
    dx[1] = v*cos(θ+β)
    dx[2] = v*sin(θ+β)
    dx[3] = v/lr*sin(β)
    dx[4] = a
    return x + dt*dx
end

"Class for vehicles"
mutable struct Car
    Type::String # "CAV" or "HDV"
    ID::Int64 # ID
    T::Float64 # Sampling time
    lf::Float64; lr::Float64 #
    st::Vector{Float64}  # Current state: x, y, θ, v
    u::Vector{Float64}  # Control inputs, a, α 

    # constant of bound constraint
    v_min::Float64; v_max::Float64
    a_min::Float64; a_max::Float64 
    α_min::Float64; α_max::Float64 

    # To save all historical data
    X_hist::Matrix{Float64}
    U_hist::Matrix{Float64}


    function Car(Type::String, ID::Int64, T::Float64, st0::Vector{Float64})
        obj = new(Type, ID, T)
        obj.st = st0
        obj.u = [0.0, 0.0]  # Control inputs
        obj.X_hist = reshape(obj.st, 4, 1)
        obj.U_hist = reshape(obj.u, 2, 1)
        return obj
    end
end

"Set the physical limit of the car"
function set_limit(c::Car, bounds, parameters)
    c.v_min = bounds["v_min"]; c.v_max = bounds["v_max"]
    c.a_min = bounds["a_min"]; c.a_max = bounds["a_max"]
    c.α_min = bounds["α_min"]; c.α_max = bounds["α_max"]
    c.lf = parameters["lf"]; c.lr = parameters["lr"]
end

"Run"
function run_lane_changing(c::Car, U; nn = 200)
    dt = c.T/nn
    a = min(c.a_max, max(c.a_min, U[1]))
    α = min(c.α_max, max(c.α_min, U[2]))
    c.u = [a, α]
    for k in 1:nn
        # Update state with the solution, and return it
        c.st = car_model(c.st, c.u, dt, [c.lf, c.lr])
    end
    c.X_hist = hcat(c.X_hist, c.st)
    c.U_hist = hcat(c.U_hist, c.u)
end

function run_car_following(c::Car, v_d; nn = 200)
    dt = c.T/nn
    for k in 1:nn
        a = min(c.a_max, max(c.a_min, PID(c, v_d)))
        α = 0.0
        c.u = [a, α]
        # Update state with the solution, and return it
        c.st = car_model(c.st, c.u, dt, [c.lf, c.lr])
    end
    c.X_hist = hcat(c.X_hist, c.st)
    c.U_hist = hcat(c.U_hist, c.u)
end

"Function to find the desired speed from constant time headway car-following model"
function CTH(c::Car, dist, CFM_params)
    τs = CFM_params["τs"];  ds = CFM_params["ds"]
    return min(c.v_max, max(c.v_min, (dist-ds)/τs)) 
end

"Find the control input to track a desired speed profile"
function PID(c::Car, vd; Kp = 1e0)
    v = c.st[4]
    return Kp*(vd-v)
end

"Compute the distance between two cars"
function distance(car_i::Car, car_j::Car)
    return sqrt((car_i.st[1]-car_j.st[1])^2 + (car_i.st[2]-car_j.st[2])^2)
end