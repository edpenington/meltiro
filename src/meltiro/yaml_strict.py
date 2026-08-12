"""A strict YAML loader for config parsing: reject duplicate mapping keys.

PyYAML's `safe_load` accepts a mapping that repeats a key and silently keeps
the last occurrence, discarding the rest during the parse. For a config file
that is the one class of malformed input the validators structurally cannot
report on: the evidence is destroyed before any validation code runs, so a
duplicated `max_checks_per_field:` in pipeline.yaml, or a field defined twice in
the template, would change the run with no error.

`strict_load` parses through `StrictLoader`, which rejects a duplicate key with
a `yaml.constructor.ConstructorError` (a `yaml.YAMLError`, exactly what
`safe_load` raises on malformed YAML). Every config parse site therefore
handles it through its own YAML-error path: the reference-list loader catches
the error and wraps it in a ConfigBundleError, the pipeline.yaml and template
loaders let it propagate as they do for any other malformed YAML.

Only config parsing uses this. Code that writes YAML or parses non-config data
is unaffected.

Known limitation: the scan covers the keys a mapping writes explicitly. Two
`<<` merge keys in one mapping are not reported; overlapping keys between the
merged sources keep PyYAML's last-wins semantics. The shipped configs use no
merge keys, so nothing depends on this today.
"""

import yaml


# YAML's merge key: `<<:` splices an aliased mapping into this one.
_MERGE_TAG = "tag:yaml.org,2002:merge"


class StrictLoader(yaml.SafeLoader):
    """A SafeLoader that rejects a key written twice in the same mapping."""

    def construct_mapping(self, node, deep=False):
        if isinstance(node, yaml.nodes.MappingNode):
            # Scan for a duplicate among the explicit keys BEFORE the base
            # loader resolves merges. Merge keys (`<<`) are skipped: PyYAML
            # prepends the merged pairs during flatten_mapping, so an explicit
            # key that overrides a merged one would otherwise collide in the
            # flattened list and be misread as a duplicate. Skipping the merge
            # entries here scans only what the source mapping actually wrote,
            # then merge resolution and the real construction are left to the
            # base loader below.
            seen = set()
            for key_node, _ in node.value:
                if key_node.tag == _MERGE_TAG:
                    continue
                key = self.construct_object(key_node, deep=deep)
                try:
                    duplicate = key in seen
                except TypeError:
                    # Unhashable key: stop scanning and let the base loader
                    # raise its own "found unhashable key" diagnostic.
                    break
                if duplicate:
                    raise yaml.constructor.ConstructorError(
                        "while constructing a mapping", node.start_mark,
                        f"found duplicate key {key!r}", key_node.start_mark)
                seen.add(key)
        return super().construct_mapping(node, deep=deep)


def strict_load(text):
    """Parse `text` (a string or file object) with `StrictLoader`.

    Drop-in for `yaml.safe_load` at a config parse site: same return value on
    valid YAML, same `yaml.YAMLError` on malformed YAML, plus a
    `ConstructorError` naming the offending key on a duplicate.
    """
    return yaml.load(text, Loader=StrictLoader)
