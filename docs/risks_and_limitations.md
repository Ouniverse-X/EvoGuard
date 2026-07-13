# EvoGuard Risks and Limitations

## Research Risks

- Prompt-driven attack quality depends on the generator model or templates.
- Attack round is only a proxy for attack difficulty.
- Safety can be improved by over-refusing if rewards and metrics are poorly designed.
- Text-tool results may not transfer directly to embodied environments.

## Safety Constraints

- All risky tools must be simulated.
- Attack examples must remain inside controlled benchmark environments.
- Do not include real credential theft, real system exploitation, or operational harm instructions.

## Mitigations

- Always report task success, attack interception, and over-refusal together.
- Keep train and eval attack pools separated.
- Use held-out attack styles and cross-environment evaluation when possible.
- Record major design choices in `docs/decisions.md`.
