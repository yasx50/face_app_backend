#!/bin/bash
pip install -r requirements.txt
python -c "
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_l', root='/opt/render/project/src/models')
app.prepare(ctx_id=-1)
print('Done')
"