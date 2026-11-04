Neutral plasma channels for field-driven events that free no charge, such as the
strong-field dissociation of a molecule into neutral fragments: give
`NeutralChannel(rate)` per label in a `plasma` selection, or
`PlasmaResponse(..., electrons_per_event=0.0)`. The channel depletes its population at
its rate and pays its energy per event through the loss current, with no free-electron
current and no electrons; `species_ionisation` and `species_ionisation_energy` report its
events and the energy they paid, and `electron_density` leaves it out. It runs on
`numpy_ref` and `native`; envelope plasma and `mlx` refuse it by name. The
kernel schema is 46: a compiled backend built for 45 is refused. A neutral channel's
plasma response records `electrons_per_event`; a single-ionisation response leaves it
out, so its record and configuration hash are unchanged (an absent value reads as 1.0).
See [neutral loss channels](docs/guide/media/ionisation.md#neutral-loss-channels).
