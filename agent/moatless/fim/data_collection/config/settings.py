class MiningConfig:
    # --- Source file extensions ---
    SOURCE_EXTENSIONS = [".py", ".java", ".ts"]

    # --- Test file patterns ---
    TEST_FILE_PATTERNS = ["test_", "_test.py", "tests/", "tests", ".spec.ts", ".test.ts"]
    FLAG = 'Single'
    if FLAG == 'Single':
        MIN_SOURCE_LOC = 1
        MAX_SOURCE_LOC = 50

        MIN_SOURCE_FILES = 1
        MAX_SOURCE_FILES = 1

        MIN_SOURCE_HUNKS = 1
        MAX_SOURCE_HUNKS = 1
    else:
        MIN_SOURCE_LOC = 5
        MAX_SOURCE_LOC = 100

        MIN_SOURCE_FILES = 1
        MAX_SOURCE_FILES = 3

        MIN_SOURCE_HUNKS = 2
        MAX_SOURCE_HUNKS = 10
    REQUIRE_DEPENDENCY = True
    ALLOW_CYCLES = False
    NO_ISOLATED_HUNKS = True

    IGNORE_FILES = ["setup.py", "__init__.py", "conftest.py", "docs/conf.py"]

    REQUIRE_TEST_CHANGE = True

    # --- LLM Configuration (use environment variables in production) ---
    LLM_API_KEY: str = "sk-placeholder-api-key"  # TODO: Replace with env var
    LLM_BASE_URL: str = "https://api.openai.com/v1"  # TODO: Replace with env var
    LLM_MODEL: str = "deepseek-chat"
    LLM_MAX_DIFF_LINES: int = 50

    USE_LLM = False