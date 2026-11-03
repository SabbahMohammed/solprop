Neutral plasma channels for field-driven events that free no charge, such as the
strong-field dissociation of a molecule into neutral fragments: give
`NeutralChannel(rate)` per label in a `plasma` selection, or
`PlasmaResponse(..., electrons_per_event=0.0)`. The channel depletes its population at
its rate and pays its energy per event through the loss current, with no free-electron
current and no electrons; `species_ionisation` and `species_ionisation_energy` report its
events and the energy they paid, and `electron_density` leaves it out. It runs on
`numpy_ref`, `numba_cpu` and `native`; envelope plasma and `mlx` refuse it by name. The
kernel schema is 42: a compiled backend built for 41 is refused, and run records gain
`electrons_per_event` on each plasma response (neutral value 1.0, so earlier records
still resume). See [neutral loss channels](docs/guide/media/ionisation.md#neutral-loss-channels).
