import importlib.metadata
import os
import platform
import sys

PACKAGES = ["mteb", "sentence-transformers", "torch", "numpy", "pandas"]

IMPORTS = [
    "from mteb.models.abs_encoder import AbsEncoder",
    "from mteb.models.model_meta import ModelMeta",
    "from mteb.types import PromptType",
]


def main():
    ok = True

    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"CPU count: {os.cpu_count()}")
    print()

    print("Installed package versions:")
    for pkg in PACKAGES:
        try:
            version = importlib.metadata.version(pkg)
            print(f"  {pkg}: {version}")
        except importlib.metadata.PackageNotFoundError:
            print(f"  {pkg}: NOT INSTALLED")
            ok = False
    print()

    print("Import checks:")
    for statement in IMPORTS:
        try:
            exec(statement)
            print(f"  OK: {statement}")
        except Exception as e:
            print(f"  FAILED: {statement}")
            print(f"    {type(e).__name__}: {e}")
            ok = False
    print()

    print("Task lookup:")
    try:
        import mteb

        task = mteb.get_task("AppsRetrieval")
        print(f"  name: {task.metadata.name}")
        print(f"  description: {task.metadata.description}")
    except Exception as e:
        print(f"  FAILED to load task 'AppsRetrieval'")
        print(f"    {type(e).__name__}: {e}")
        ok = False
    print()

    if not ok:
        print("check_env: FAILED")
        sys.exit(1)

    print("check_env: OK")


if __name__ == "__main__":
    main()
