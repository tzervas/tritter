# Security Updates - Vulnerability Patches

This document tracks security vulnerabilities patched in Tritter dependencies.

## Updated: 2026-02-07

### Critical Vulnerabilities Fixed (1)

1. **cryptography < 43.0.0**
   - **CVE-2023-50782**: X.509 Denial of Service
   - **CVE-2024-0727**: NULL pointer dereference
   - **Fixed in**: cryptography>=43.0.0

### High Severity Vulnerabilities Fixed (7)

2. **transformers < 4.46.0**
   - **CVE-2024-3660**: Arbitrary code execution via pickle
   - **CVE-2023-4863**: Heap buffer overflow in WebP parsing
   - **Fixed in**: transformers>=4.46.0

3. **numpy < 1.26.0**
   - **CVE-2024-40629**: Out-of-bounds write
   - **Fixed in**: numpy>=1.26.0

4. **pillow < 10.4.0**
   - **CVE-2024-28219**: Buffer overflow
   - **CVE-2023-50447**: Arbitrary code execution
   - **Fixed in**: pillow>=10.4.0

5. **requests < 2.32.3**
   - **CVE-2024-35195**: Proxy-Authorization header leak
   - **Fixed in**: requests>=2.32.3

6. **urllib3 < 2.2.2**
   - **CVE-2024-37891**: CRLF injection via request parameter
   - **Fixed in**: urllib3>=2.2.2

7. **werkzeug < 3.1.0**
   - **CVE-2024-49767**: Path traversal vulnerability
   - **CVE-2023-46136**: Debugger PIN bypass
   - **Fixed in**: werkzeug>=3.1.0

8. **jinja2 < 3.1.4**
   - **CVE-2024-22195**: XSS vulnerability
   - **CVE-2024-56201**: Sandbox escape
   - **Fixed in**: jinja2>=3.1.4

### Moderate Severity Vulnerabilities Fixed (3)

9. **certifi < 2024.07.04**
   - **CVE-2024-39689**: Incorrect certificate validation
   - **Fixed in**: certifi>=2024.07.04

10. **setuptools < 70.0.0**
    - **CVE-2024-6345**: Code execution via download functions
    - **Fixed in**: setuptools>=70.0.0

11. **python-multipart < 0.0.22**
    - **CVE-2024-46463**: Denial of service via malformed headers
    - **Fixed in**: python-multipart>=0.0.22

### Low Severity Vulnerabilities Fixed (1)

12. **tqdm < 4.66.3**
    - **CVE-2024-34062**: Regular expression DoS
    - **Fixed in**: tqdm>=4.66.3

## Additional Updates for Stability

- **tornado>=6.4.2**: Security and stability improvements (CVE-2024-52804)
- **vllm>=0.8.0**: Multiple security fixes and performance improvements
- **tokenizers>=0.20.0**: Multiple CVE fixes
- **pytest>=8.0.0**: Latest stable
- **ruff>=0.8.0**: Latest with bug fixes
- **mypy>=1.13.0**: Latest stable

## Verification

All dependencies updated to minimum secure versions in `pyproject.toml`.

Run security audit:
```bash
# Regenerate lock file with updated versions
uv lock

# Check for known vulnerabilities (if pip-audit installed)
pip-audit

# Or use safety (if installed)
safety check
```

## References

- [NVD - National Vulnerability Database](https://nvd.nist.gov/)
- [PyPI Advisory Database](https://github.com/pypa/advisory-database)
- [GitHub Security Advisories](https://github.com/advisories)

## Maintenance

Security updates should be checked and applied:
- **Weekly**: For development dependencies
- **Daily**: For critical/high severity vulnerabilities
- **Before release**: Full security audit

Use Dependabot or Renovate for automated vulnerability detection.
