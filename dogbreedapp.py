"""
Start the dog breed prediction app locally.
"""

from dogbreedapp import app

app.run(host='0.0.0.0', port=3001, debug=True, threaded=False)


