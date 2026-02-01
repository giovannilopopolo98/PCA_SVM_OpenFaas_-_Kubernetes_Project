import json
import pandas as pd
from .core import main_logic, BadRequest

def handle(req):
    try:
        data = json.loads(req)
        if "data" not in data:
            raise BadRequest("Chiave 'data' mancante nel JSON.")

        df = pd.DataFrame(data["data"])
        result = main_logic(df)
        return json.dumps({"result": result}, ensure_ascii=False)

    except BadRequest as e:
        return json.dumps({"error": str(e)})

    except Exception as e:
        return json.dumps({"error": f"Errore interno: {str(e)}"})
