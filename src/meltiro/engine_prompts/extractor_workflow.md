The extraction record is built incrementally through tool calls. Field-level guidance (descriptions, allowed values, evidence requirements) lives in the tool input schemas attached to this request.

**First call: `record_initial_check`.** Until it lands, every call that would change the extraction output is refused. It reports on the material the extractor was handed *before* anything is extracted from it, so it has to come first to mean anything. Its properties are the initial-check variables themselves, passed flat as **bare values** (a string, a boolean, a list of strings) with no `{value, evidence, notes}` envelope: they describe the extraction process rather than paper content, so evidence does not apply. It may be called again to revise it.

**Then the extraction.** Call `update_study` with every populated study field, and `add_record` once per distinct record the paper reports. The `add_record` schema names what one record stands for; this review's own criteria decide which ones qualify. These can go in a single response carrying many `tool_use` blocks.

**Validation feedback.** Each call is deterministically validated per field. The result reports `status: ok` when every field applied, `partial` when some applied and others failed (only the failed fields, listed under `failed_fields`, need resubmitting), or `validation_failed` when nothing applied.
