Notes are separate from evidence, and neither kind is validated: a note is commentary, not a claim about the paper.

- **Field notes.** Every envelope field carries a `notes` slot beside its `value` and `evidence`: how a number was read off a table, why one of several reported estimates was chosen, what made a judgement finely balanced. The checker reads it. It never substitutes for required evidence.
- **Scope notes.** `update_study`, `add_record`, and `update_record` each take an optional top-level `notes` argument holding one free-text note about that whole scope. The checker is not given these, so anything a specific field's value depends on goes in that field's notes instead.
