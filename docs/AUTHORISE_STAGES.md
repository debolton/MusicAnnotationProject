# Authorise stages
VALIDATE_INPUT → SAVE_AUTHORITY → PREPARE_MASK → TELEA → GEOMETRY_T80 → SAVE_ROUTING → COMPONENTS → SAVE_PREDICTORS → VERIFY_V77F → SCORE_V77F → SERIALISE_RESPONSE → COMPLETE

`authorise_status.json` is rewritten before every stage. On failure, `authorise_error.txt` contains the traceback and the API reports stage, exception type and exception text.
