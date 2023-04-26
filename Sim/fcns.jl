"Some neccesary functions"
function sigmoid(x)
    return 1/(1 + exp(-x))
end

function sigmoid_der(x)
    return sigmoid(x)*(1-sigmoid(x))
end

# Function to generate random entering time
function randomTimeGen(number::Int64; Timestep = 0.1, mu = 5.0, sigma = 1.0, offset = 3.0)
    t_diff = rand(Normal(mu, sigma), number) 
    t_diff = max.(t_diff, offset)
    tt = cumsum(t_diff)
    return Timestep*ceil.(tt/Timestep)
end