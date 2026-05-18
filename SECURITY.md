# Reporting a Vulnerability

To report a security vulnerability, please email boswell.labs@gmail.com.

We take security seriously and will respond to security reports within 48 hours. Please include as much detail as possible about the vulnerability, including:

- A description of the vulnerability
- Steps to reproduce the issue
- Potential impact
- Suggested fix (if any)

While the discovery of new vulnerabilities is rare, we also recommend always using the latest version of dspy-lancedb-memory to ensure your application remains as secure as possible.

## Security Considerations for dspy-lancedb-memory

As dspy-lancedb-memory is a library that processes, embeds, and stores conversation data in a vector database, please be aware of the following security practices:

- **LLM Extraction**: dspy-lancedb-memory sends conversation content to an LLM (configured via `extraction_lm`) for memory extraction. Ensure you trust the LLM provider and that sensitive data is handled according to your organization's data governance policies.
- **Data Storage**: Memory data (including extracted facts from conversations) is persisted in a LanceDB database on local storage. Protect the database file and directory with appropriate filesystem permissions, especially if storing sensitive information.
- **Embedding & Reranking APIs**: Embedding and reranking requests are sent to external API providers as configured. Review each provider's data handling policies before sending production or sensitive data.
- **Prompt Injection**: Memory extraction uses DSPy signatures with LLM prompts. Be aware that crafted conversation input could influence extraction output. Validate or sanitize stored memories if they are used in security-sensitive contexts.

## Security Hall of Fame

We would like to thank the following security researchers for responsibly disclosing security issues to us.

*No security researchers have been added to the hall of fame yet. Will you be the first?*
