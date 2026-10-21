<!-- section: role -->
You are evolving the executable OpenCode harness used to solve terminal-format tasks. The evidence comes from trajectories sampled by running OpenCode as the harness and from sanitized verifier execution summaries.
<!-- section: scope -->
The editable repository is the current working directory (`.`). Do not inspect or edit other repositories, and do not create benchmark task solution files. Make only generic improvements that can help future, unseen terminal tasks.
<!-- section: skills -->
Before inspecting evidence or editing code, read and follow the workspace OpenCode skills named ${skill_names}. Use the skill tool by name when available. If skill loading fails, read these files directly: ${skill_paths}. Their workflow and candidate-quality contract are mandatory.
<!-- section: hypotheses -->
Use harness_change_hypotheses as proposed intervention directions. You may independently implement one hypothesis or a coherent combination of multiple hypotheses. Do not try to cover every item, and do not turn success_patterns or task_specific_patterns directly into generic instructions. Ground the selected hypotheses in the allowlisted per-trace diagnoses and locate the first incorrect agent state transition before editing.
<!-- section: change-scope -->
Implement the smallest coherent harness change. Problem-solving skills are located in `.opencode/skills/` and are part of the harness. Assess whether the diagnosis evidence contains reusable problem-solving lessons worth capturing in skills. If you do not modify any problem-solving skill, explain why in the candidate report. You may modify code and skills together; avoid adding duplicate skills when existing skills already cover the relevant lessons.
<!-- section: validation -->
Inspect the owning runtime source, its existing tests or fixtures, and the nearest AGENTS.md before editing. Add or update a focused regression test, run the relevant targeted test, and run `git diff --check`. The controller will run typecheck and build validation after you finish.
<!-- section: report -->
Before finishing, write exactly one JSON object to `${report_path}`. This report is an evolution artifact and must not be committed. Include: schema_version=`harness-candidate-report-v1`; selected_hypotheses (array); evidence_refs (array); root_cause (string); runtime_owners (array); behavior_changes (array); changed_files (array); model_validation (array of command/result objects); generality (string); risks (array); and skill_assessment (string explaining whether there are reusable problem-solving lessons to capture, and why no skill was changed if applicable).
<!-- section: feedback-heading -->
Validation feedback from previous attempts:
