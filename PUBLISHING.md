# Publishing `pdfagent-rag` to PyPI

This repo publishes to PyPI as **`pdfagent-rag`** (the bare `pdfagent` name on
PyPI is owned by an unrelated project). The console command shipped by the
package is still `pdfagent`.

There are two ways to ship a new release: a one-time **automated** path via
GitHub Actions (recommended), and a fully **manual** path you can fall back to.

---

## One-time setup — PyPI Trusted Publishing (OIDC)

This is the modern, token-free way to publish. GitHub Actions authenticates to
PyPI via OpenID Connect, so you never paste an API token into a secret.

1. **Create the PyPI account** at https://pypi.org/account/register/ if you
   don't have one yet, and verify your email.
2. **Reserve the project name** by uploading the first release manually (see
   the [Manual fallback](#manual-fallback) below). PyPI only lets you configure
   trusted publishing on projects that already exist.
3. **Add the GitHub publisher** at https://pypi.org/manage/project/pdfagent-rag/settings/publishing/ →
   *Add a new pending publisher*:
   - **Owner:** `bhagatdas`
   - **Repository:** `PdfAgent-Agentic-RAG-over-any-PDF`
   - **Workflow:** `publish.yml`
   - **Environment:** `pypi`
4. **Create the GitHub Environment** in the repo: *Settings → Environments →
   New environment → name `pypi`*. No secrets to add. (Optional: require
   reviewer approval before publish runs.)

After that, every `git tag vX.Y.Z && git push origin vX.Y.Z` triggers a build +
publish via `.github/workflows/publish.yml` — no token, no twine, no manual
steps.

---

## Cutting a release

```bash
# 1. Bump version in pyproject.toml (and commit)
# 2. Tag
git tag v0.1.0
git push origin v0.1.0
# 3. GitHub Actions builds + publishes; watch at
#    https://github.com/bhagatdas/PdfAgent-Agentic-RAG-over-any-PDF/actions
```

When the run finishes, `pip install pdfagent-rag==0.1.0` works for everyone.

---

## Manual fallback

If you ever need to publish without GitHub Actions (or you're doing the very
first release to reserve the name):

```bash
# Clean previous build artifacts
rm -rf dist/ build/ *.egg-info

# Build sdist + wheel
python -m pip install --upgrade build twine
python -m build

# Optional: smoke-test the wheel in a fresh venv before uploading
python -m venv /tmp/_pdfagent_check && /tmp/_pdfagent_check/bin/pip install dist/*.whl

# Upload (will prompt for username "__token__" + a PyPI API token —
# create one at https://pypi.org/manage/account/token/, scoped to this project)
python -m twine upload dist/*
```

For a dry run, upload to TestPyPI first:

```bash
python -m twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ pdfagent-rag
```

---

## Version bumps

`pyproject.toml` has `version = "0.1.0"` hard-coded. Bump it before tagging.
Follow SemVer: patch (0.1.x) for bug fixes, minor (0.x.0) for new features,
major (x.0.0) for breaking changes. Tag names must match (`v0.1.1`, `v0.2.0`).

If you forget to bump and re-tag the same version, PyPI will reject the upload
— versions are immutable once published.
