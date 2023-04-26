# push!(LOAD_PATH, ".")
using Pkg; Pkg.activate("."); Pkg.instantiate()
using Plots, GaussianProcesses, Random, DelimitedFiles

include("fcns.jl")
include("path.jl")
include("car.jl")
include("controller.jl") 
# include("estimator.jl")
# include("util.jl")

T = 0.2; H = 20; G = 20
solver = "Ipopt"

Random.seed!(16);
# velocity and input bounds
bounds = Dict("v_min" => 0.0, "v_max" => 30.0, "a_min" => -3.0, "a_max" => 2.0, "α_min" => -π/12, "α_max" => π/12)
parameters = Dict("lf" => 1.03, "lr" => 1.54)

## Initialize the car objects
yc_i = -6.0; yc_f = 0.0
vd = 28.0; τs = 2.0; ds = 8.0

CAV_1 = Car("CAV", 1,  T, [0.0, yc_i, 0.0, 28.4])
HDV_2 = Car("HDV", 2, T, [30.0, yc_f, 0.0, 28.0])
HDV_3 = Car("HDV", 3, T, [-30.0, yc_f, 0.0, 26.6])
Cars = [CAV_1, HDV_2, HDV_3]
for car in Cars
    set_limit(car, bounds, parameters)
end

control = MPC(T, H)
set_limit(control, bounds, parameters)
set_ref(control, yc_f, vd)
set_state(control, CAV_1, HDV_3)
set_nominal(control, zeros(2, H))
W_AH = 1e3
W_H2 = 10.0 .^[0.0, 1.0]
W_H3 = 10.0 .^[-1.0, 0.0]
weights = Dict("Wu" => [1e-1, 1e1], "Wv" => 1e-1, "Wy" => 1e-3, "Wd" => W_AH, "λ" => 1e9,
        "WHu" => W_H3[1], "WHv" => W_H3[2],
        "y_min" => yc_i, "y_max" => yc_f, "Δθ_min" => -5/180*π, "Δθ_max" => 5/180*π, "ϵ" => 1e-5)
set_params(control, weights)

# nonlinearMPC(control, vd)
linearizedMPC(control, vd)

L = 100
t_comp = []
for t in 1:L
    println("Time step ", t)

    set_state(control, CAV_1, HDV_3)

    # Run HDV_2 using IRL-CFM model
    u_HDV_2 = input_for_HDV(Cars, 2, 1, vd, [W_H2; W_AH])
    run_car_following(HDV_2, u_HDV_2*T + HDV_2.st[4])

    # Run HDV_3 using IRL-CFM model
    vd_3 = CTH(HDV_3, HDV_2.st[1] - HDV_3.st[1], τs, ds)
    if distance(HDV_2, HDV_3) < distance(CAV_1, HDV_3) j = 2 else j = 1 end
    u_HDV_3 = input_for_HDV(Cars, 3, j, vd_3, [W_H2; W_AH])
    run_car_following(HDV_3, u_HDV_3*T + HDV_3.st[4])

    # Run CAV using MPC
    if HDV_3.st[1] - CAV_1.st[1] > 0.0 hw = HDV_3.st[1] - CAV_1.st[1] else hw = HDV_2.st[1] - CAV_1.st[1] end
    v_cfm = CTH(CAV_1, hw, τs, ds)
    set_ref(control, yc_f, v_cfm)
    U, solving_time = nonlinearMPC(control, vd_3)
    # U, solving_time = linearizedMPC(control, vd_3)
    append!(t_comp, solving_time)
    run_lane_changing(CAV_1, U[:,1])
end


# Plot the results
T_hist = [T*i for i in 1:L+1]

gr()
plot(CAV_1.X_hist[1,:], CAV_1.X_hist[2,:], color=:red)

plot()
for car in Cars
    car.Type == "CAV" ? c = :green : c = :red
    display(plot!(T_hist, car.X_hist[1,:], color=c))
end

plot(T_hist, CAV_1.X_hist[3,:])
plot(T_hist, CAV_1.X_hist[4,:])
plot(T_hist, HDV_2.X_hist[4,:])
plot(T_hist, HDV_3.X_hist[4,:])
dist = sqrt.((HDV_3.X_hist[1,:]-CAV_1.X_hist[1,:]).^2 + (HDV_3.X_hist[2,:]-CAV_1.X_hist[2,:]).^2)
plot(T_hist, dist)
minimum(dist)
plot(T_hist, CAV_1.U_hist[2,:])
plot(t_comp)


## Look-up table to find the optimal control weights
grid_size = (9,9)
nodes = (Vector(LinRange(-2.0, 2.0, grid_size[1])), Vector(LinRange(-2.0, 2.0, grid_size[2])))
TABLE = readdlm("./input/Time-Energy/Dec24_weight_table_81_TE.csv", ',')

data = TABLE'
x = data[3:4,:];              #predictors
y1 = data[1,:];   #regressors
#Select mean and covariance function
mZero = MeanZero()                   #Zero mean function
kern = SE(0.0,0.0)                   #Sqaured exponential kernel (note that hyperparameters are on the log scale)
logObsNoise = -5.0                        # log standard deviation of observation noise (this is optional)
gp1 = GP(x, y1, mZero, kern, logObsNoise)       #Fit the GP
GaussianProcesses.optimize!(gp1)                         # Optimize the hyperparameters
μ1, σ² = predict_y(gp1, x);
print(maximum(y1 - μ1))
# plot(heatmap(gp1); fmt=:png)

y2 = data[2,:];   #regressors
#Select mean and covariance function
mZero = MeanZero()                   #Zero mean function
kern = SE(0.0,0.0)                   #Sqaured exponential kernel (note that hyperparameters are on the log scale)
logObsNoise = -5.0                        # log standard deviation of observation noise (this is optional)
gp2 = GP(x, y2, mZero, kern, logObsNoise)       #Fit the GP
GaussianProcesses.optimize!(gp2)                         # Optimize the hyperparameters
μ2, σ² = predict_y(gp2, x);
print(maximum(y2 - μ2))
# plot(heatmap(gp2); fmt=:png)

gp = (gp1, gp2)

## Load map and initialize a list of path objects
segment_dict, path_dict, node_dict = load_data()

path1 = Path(1, 100, 80)
path2 = Path(2, 100, 80)
paths = [path1, path2]


W_H = 10 .^[-1.0, -0.5]


## Run single simulation to get the trajectory
P_hist, V_hist, U_hist, comp_time, cost, safe = main_BO_GP(gp, paths, 10 .^[-1.0, -0.5];
                                                    p0 = 5.0, v0 = 9.0, solver = "Ipopt", H = 20)

# Plot the results
using Plots
plot(P_hist')
plot(V_hist')
plot(U_hist[1,:])
plot(comp_time)

# Save data for later animation in python
# prefix = string(Dates.monthname(today())[1:3], Dates.day(today()))
# file_name = string("./results/", prefix, "_trajectory.csv")
# U_hist = hcat(U_hist, [NaN, NaN])
# DATA = vcat(P_hist, V_hist, U_hist)'
# writedlm(file_name, DATA)

## Comparison between BayOpt and SVO
BO_stats = Dict("SAFE" => [], "COST" => [])
SVO_stats = Dict("SAFE" => [], "COST" => [])
seed = 111; Random.seed!(seed)
n_simulation = 200

for i in 1:n_simulation
    println("Simulation ", i)
    p0 = rand(Uniform(-10.0, 10.0))
    v0 = rand(Uniform(5.0, 12.0))
    W_A = 10 .^ [rand(Uniform(-2.0, 2.0)), rand(Uniform(-2.0, 2.0))]
    _, _, _, _, cost1, _ = main_BO_GP(gp, paths, W_A; p0 = p0, v0 = v0, solver = "Ipopt", H = 10)
    _, _, _, _, cost2, _ = main_SVO(paths, W_A; p0 = p0, v0 = v0, solver = "Ipopt", H = 10)

    push!(BO_stats["COST"], cost1)
    push!(SVO_stats["COST"], cost2)

    if i%10 == 0
        # Number of simulations with safe guarantee
        println("Number of safe simumations: ", count(BO_stats["COST"] .< 1e3), ", ", count(SVO_stats["COST"] .< 1e3))

        idx = [i for i in 1:length(BO_stats["COST"]) if BO_stats["COST"][i] < 1e3 && SVO_stats["COST"][i] < 1e3]

        # Number of simulations with improvements
        n_improv = count(SVO_stats["COST"][idx] .> BO_stats["COST"][idx])/length(idx)
        println("Percentages of improvements: ", n_improv)

        # Average percentage of improvements
        aver_improv = mean((SVO_stats["COST"][idx] - BO_stats["COST"][idx])./SVO_stats["COST"][idx])
        println("Average improvements: ", aver_improv)
    end
end

# Number of simulations with safe guarantee
println("Number of safe simumations: ", count(BO_stats["COST"] .< 1e3), ", ", count(SVO_stats["COST"] .< 1e3))

idx = [i for i in 1:length(BO_stats["COST"]) if BO_stats["COST"][i] < 1e3 && SVO_stats["COST"][i] < 1e3]

# Number of simulations with improvements
n_improv = count(SVO_stats["COST"][idx] .> BO_stats["COST"][idx])/length(idx)
println("Percentages of improvements: ", n_improv)

# Average percentage of improvements
aver_improv = mean((SVO_stats["COST"][idx] - BO_stats["COST"][idx])./SVO_stats["COST"][idx])
println("Average improvements: ", aver_improv)

println("THAT IS THE END !!!")


## Save data for later animation in python
# using Dates
# prefix = string(Dates.monthname(today())[1:3], Dates.day(today()))
# file_name = string("./results/", prefix, "_statistics_", seed, ".csv")
# STATS = hcat(BO_stats["COST"], SVO_stats["COST"])
# writedlm(file_name, STATS)



coeffs = [1, -2, -2, 1]
@elapsed roots(Polynomial(coeffs))
@elapsed find_zero(Polynomial(coeffs), (-1e3, 1e3), Bisection())[end]