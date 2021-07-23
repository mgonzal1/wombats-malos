# WOMBAT Project

Development of an end-to-end (stim-to-action) neural network trained only with an encoder (stim->sensory representation) and a decoder (motor choice representation -> action). The intermediate mapping can be trained with additional brain representations, or with a mapping between sensory and motor representations. 

The goal is to create a testbed model trained on real neural data, to test hypotheses ranging from number of neurons needed for computation, plasticity, timing, and the utility of other brain regions in the stim-to-action behavior. Because the model is never explicitly trained on stim-to-action, the internal representations and computations of the model can be compared to real neural data without any model bias.

__Model__

![Alt text](/img/model.png?raw=true "Title")


## To install:
```
$ cd your_path
$ git clone https://github.com/mgonzal1/wombats-malos.git
$ pip install -e wombats
```

## Importing
```
import wombats
# or to import specific modules
from wombats import metrics
from wombats import models

# if you change the source files, and want to test the results in a console:
from importlib import reload
metrics = reload(metrics)
models = reload(models)
```

### Check the notebooks directory for initial usage



More about the authors 

Adriana Pliego, 
Francisca Mart ́ınez Traub, 
Natasha de la Rosa, 
Naybi Nikte-Ha Requejo-Mendoza,
Mildred Salgado M enez, 
Mariana Gonzalez Velarde, 
Jonaz Moreno Jaramillo, 
Alexander Gonzalez

It was the first months of 2020, and the words "pandemic", "virus" and "quarantine" were on everybody's screens. Then, around the world, people started creating new ways to connect again to escape from isolation. The scientific community migrated conferences from expensive locations to zoom meetings, and people like the neuroscientist Konrad Kording, gathered in Twitter a community of enthusiast scientists with the common interest of teaching computational neuroscience methods. This "neuromatch" made in heaven had the purpose of making a synchronous computational neuroscience summer school for people from all corners of the world (an ambitious goal) and all kinds of backgrounds. From data science to medicine, from Cairo to Mexico. All together in one breakout room. 

So, "Neuromatch Academy"  was born in August 2020, where we found each other in the "Massive Wombats" pod zoom session in our first meeting with our TA, Alex Gonzalez. During these three weeks, we worked through tutorials and on our projects, exploring datasets such as the Steinmetz dataset. We got interested in how the encoding of a stimulus results in a motor command. We then asked, "how the placement of in silico lesions in different brain areas modify this process?".  We were birds of the same feather which in the end resulted in a group of people enjoying sharing their experiences in different countries and learning together. 

