"Class for paths"
mutable struct Path
    ID::Int64  
    length::Float64 # total length of the CZ
    conflict # Position of conflict point wrt entry

    function Path(ID, conflict, length)
        obj = new(ID, conflict, length)
        return obj
    end
end
