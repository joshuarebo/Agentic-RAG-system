import os
import tempfile

# Set up test environment BEFORE any app code is imported.
# conftest.py is loaded by pytest before test modules, so env vars
# are available when app/config.py reads them via pydantic-settings.
os.environ.setdefault("OPENROUTER_API_KEY", "test-key-not-real")
os.environ.setdefault("CHROMA_PERSIST_DIR", tempfile.mkdtemp())
