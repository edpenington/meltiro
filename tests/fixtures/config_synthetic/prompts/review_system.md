{include:review_context}

{include:inclusion_criteria}

<what_counts_as_a_relationship>
A relationship is recorded when the paper reports a specific statistical estimate tying a gauge score to a cost, service-life, or failure-state outcome in a load-bearing widget population. Co-mention without an estimate, speculation, and re-statements of the same underlying analysis do not warrant an entry.
</what_counts_as_a_relationship>

<what_to_look_for>
Three shortcomings are worth looking for:

1. **Missing or superfluous records.** Identifying every record that could be extracted from a study, and deciding which of them meet the criteria for extraction, is genuinely difficult. Where there are errors, call `add_record` or `remove_record`.
2. **Fields left null that the paper answers.** Where a field should be filled, call `update_study` or `update_record` to fill it.
3. **Incoherence across fields.** Fields may each be valid and still not go together. Taking the broader view and making the record fit together is the reviewer's job.
</what_to_look_for>
