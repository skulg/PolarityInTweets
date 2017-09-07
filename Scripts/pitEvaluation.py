# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%run 'pit.ipynb'

# <codecell>

makeDicts()
makeSets()
classifier,classifier2,clf=trainClassifiers()

# <codecell>

showLv1stats()

# <codecell>

scorerSVM(goldSetTAX,goldSetTAY,clf)
scorerSVM(goldSetSAX,goldSetSAY,clf)
scorerSVM(goldSetTBX,goldSetTBY,clf)
scorerSVM(goldSetSBX,goldSetSBY,clf)

# <codecell>

