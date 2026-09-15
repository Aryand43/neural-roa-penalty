using Logging

# Counts OptimizationOptimisers' "Skipping parameter update due to NaN or Inf in gradients"
# warnings (the update is dropped, not applied) and forwards every other log message.
mutable struct GradSkipCounter <: AbstractLogger
    parent::AbstractLogger
    count::Int
end
GradSkipCounter(parent::AbstractLogger = current_logger()) = GradSkipCounter(parent, 0)

const GRAD_SKIP_PATTERN = "NaN or Inf in gradients"

Logging.min_enabled_level(l::GradSkipCounter) = min(Logging.Warn, Logging.min_enabled_level(l.parent))
Logging.shouldlog(::GradSkipCounter, args...) = true
Logging.catch_exceptions(l::GradSkipCounter) = Logging.catch_exceptions(l.parent)

function Logging.handle_message(l::GradSkipCounter, level, message, _module, group, id,
        file, line; kwargs...)
    if occursin(GRAD_SKIP_PATTERN, string(message)) ||
            any(v -> occursin(GRAD_SKIP_PATTERN, string(v)), values(kwargs))
        l.count += 1
        return nothing
    end
    Logging.shouldlog(l.parent, level, _module, group, id) || return nothing
    return Logging.handle_message(l.parent, level, message, _module, group, id, file, line;
        kwargs...)
end
