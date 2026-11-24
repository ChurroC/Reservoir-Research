---
id: default
name: Default
version: 1.0.0
author: ZotLit
description: The built-in format for literature notes.
contract: 3
filename: |
  {{ zt.citationKey | default: zt.DOI | default: zt.title | default: zt.key }}{% suffix %}
language: liquid
frontmatter:
  - key: title
    expr: zt.title
    merge: replace
  - key: related
    expr: zt.relatedItems | note_links
    merge: replace
  - key: collections
    expr: zt.collections | collection_paths
    merge: replace
  - key: citekey
    expr: zt.citationKey
    merge: replace
---
# {{ zt.title }}

[Zotero]({{ zt.backlink }}){% if zt.DOI %} | [Paper](https://doi.org/{{ zt.DOI }}){% elsif zt.url %} | [Paper]({{ zt.url }}){% endif %}

## Annotations

```zotquery
views:
  - type: annotations
    name: Annotations
    from: annotations
    filters: item == this.item
    sort:
      - property: attachment.dateAdded
        direction: ASC
      - property: sortIndex
        direction: ASC
```
--- zotlit:annotation ---
{% bq %}
[!note] {% if zt.pageLabel and zt.pageLabel != "" %}Page [{{ zt.pageLabel }}]({{ zt.backlink }}){% else %}[Zotero]({{ zt.backlink }}){% endif %}

{{ zt.imgLink | embed }}{{ zt.text }}{% if zt.citation %} {{ zt.citation }}{% endif %}
{% if zt.comment %}

{{ zt.comment }}
{% endif %}
{% endbq %}
