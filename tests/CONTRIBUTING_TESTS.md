# Adding New Tests

When adding a new test, place it in the correct subsystem:

- Gate behavior → `tests/cpu/gates/`
- Drift, saturation, stability → `tests/cpu/drift/`
- Policy forward behavior → `tests/cpu/policy/`
- PPO algorithm → `tests/cpu/ppo/`
- Infrastructure → `tests/cpu/infra/`

## Guidelines

1. **Prefer invariants over exact numbers.**  
   Use relationships (>=, <=, monotonicity) instead of fixed values.

2. **Use tolerances for float32.**  
   Typical epsilon: `1e‑5`.

3. **Use expectation‑based tests for stochastic behavior.**  
   Average over multiple random sequences.

4. **Keep tests fast.**  
   CPU tests should complete in < 5 seconds.

5. **Document the invariant.**  
   Every test file should include a docstring explaining the mathematical property being validated.