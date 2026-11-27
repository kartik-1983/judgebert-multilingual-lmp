PreserveLex support files

Copy the two Python files from this archive's src folder into your existing
C:\judgebert-multilingual-lmp\judgebert_multilingual\src folder.

Files:
  src\__init__.py
  src\legal_data.py
  src\legal_checks.py

Important:
- These files resolve the missing-module imports so the restored UI can start.
- legal_data.py contains only a minimal India placeholder and no fabricated law
  entries. The original curated legal library data must be restored separately.
- legal_checks.py contains transparent keyword heuristics only. It is not a
  trained classifier, legal analysis engine, or validated legal detector.
- The model checkpoint and src\model.py are not changed.
