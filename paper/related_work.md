# Related Work Draft

## Automated Jailbreak Red Teaming

PAIR (Prompt Automatic Iterative Refinement) uses an attacker LLM to iteratively refine black-box jailbreak prompts against a target model, often finding successful jailbreaks in fewer than twenty target queries. TAP (Tree of Attacks with Pruning) extends this line with tree search and pruning, using an attacker model to retain promising branches while reducing target queries. GPTFuzzer frames jailbreak discovery as fuzzing: it starts from seed templates, mutates them, and uses a judge to select successful candidates.

These works make the current EvoGuard red-team story stronger but also raise the bar for evaluation. EvoGuard should not claim novelty from merely generating prompt-injection templates. The distinction should be that PAIR/TAP/GPTFuzzer mainly optimize harmful-response jailbreak prompts for chat models, while EvoGuard studies evolving attacks against pre-execution tool-call safety decisions with task success, over-refusal, and attribution measured together.

Implementation note: the codebase includes a controlled `automated_jailbreak` split that maps PAIR, TAP, and GPTFuzzer ideas into simulated tool-use attacks. These examples remain fake external-context conflicts rather than unrestricted harmful-content prompts.
