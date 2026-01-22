# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) for the Tritter project.

## What is an ADR?

An Architecture Decision Record (ADR) documents a significant architectural decision along with its context and consequences. ADRs help us:

- Understand why certain design choices were made
- Provide context for future contributors
- Track the evolution of architectural thinking
- Avoid revisiting already-decided questions

## ADR Format

Each ADR follows this structure:

1. **Status**: Proposed, Accepted, Deprecated, Superseded
2. **Context**: What is the issue we're addressing?
3. **Decision**: What are we doing about it?
4. **Consequences**: What becomes easier or harder as a result?
5. **Alternatives Considered**: What other options did we evaluate?

## Index of ADRs

### Active

- [ADR-001: Sequence Position vs Token Semantics](./001-sequence-position-vs-token-semantics.md) - Defines how we measure sequence length in embedding-prediction models

### Template

When creating a new ADR, use this template:

```markdown
# ADR XXX: [Title]

**Status**: [Proposed|Accepted|Deprecated|Superseded]
**Date**: YYYY-MM-DD
**Decision Makers**: [Who was involved]
**Tags**: [relevant, tags]

## Context and Problem Statement

[Describe the issue and why a decision is needed]

## Decision

[The decision that was made]

### Rationale

[Why this decision was chosen]

## Consequences

### Positive
- [Good outcomes]

### Negative
- [Drawbacks or new challenges]

### Neutral
- [Outcomes that are neither clearly good nor bad]

## Alternatives Considered

### Alternative 1: [Name]
**Rationale**: [Why we considered this]
**Rejected because**: [Why we didn't choose it]

## References

1. [Relevant links]
```

## Contributing

When making significant architectural decisions:

1. Create a new ADR in this directory with the next sequential number
2. Use the template above
3. Include thorough analysis of alternatives
4. Link to relevant research papers, code, or documentation
5. Update this README's index
6. Submit for review before implementing

## Further Reading

- [Architecture Decision Records (ADR) Pattern](https://adr.github.io/)
- [When to Write an ADR](https://github.com/joelparkerhenderson/architecture-decision-record#when-to-write-an-adr)
